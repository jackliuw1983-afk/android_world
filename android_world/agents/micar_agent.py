# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter for the micar UI automation agent.

Wraps the micar agent (an external Claude-API + Midscene CLI agent that drives a
device directly via adb) as an AndroidWorld ``EnvironmentInteractingAgent``.

The micar agent runs its own full action loop end-to-end inside a single
``step`` call: it connects to the device, screenshots, and iterates until it
signals ``finish``. AndroidWorld then computes the reward with its own
ground-truth verifier (``task.is_successful``), so ``done`` is always ``True``
regardless of the subprocess exit code -- the device state is the source of
truth, not the agent's own finish signal.

When ``output_dir`` is set, the subprocess's combined stdout/stderr is tee'd to
both a per-task log file under ``output_dir`` and this process's stdout, and the
``MIDSCENE_REPORT_OUTPUT_DIR`` env var is injected so micar copies its Midscene
HTML report there. This is controlled by ``run.py``'s ``--agent_output_dir``
flag, so no shell-level env-var edits are needed.
"""

import json
import os
import re
import shlex
import subprocess
import sys

from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface


class MicarAgent(base_agent.EnvironmentInteractingAgent):
  """Runs the micar agent on AndroidWorld tasks via a subprocess."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      adb_serial: str = 'emulator-5554',
      command: str | None = None,
      model: str | None = None,
      max_turns: int | None = None,
      thinking_mode: str | None = None,
      use_a11y: bool = False,
      output_dir: str | None = None,
  ):
    super().__init__(env, name='micar_agent', transition_pause=None)
    self._adb_serial = adb_serial
    # Command to invoke the micar CLI. Override with the MICAR_AGENT_CMD env var
    # (e.g. "/path/to/.venv/bin/python /path/to/micar/main.py") before the
    # packaged binary exists. shlex.split lets the value carry its own args.
    command = command or os.environ.get('MICAR_AGENT_CMD', 'micar-agent')
    self._base_cmd = shlex.split(command)
    self._model = model
    self._max_turns = max_turns
    # Thinking mode (enabled/disabled) injected as THINKING_MODE into the micar
    # subprocess env. None leaves micar's own default untouched.
    self._thinking_mode = thinking_mode
    # Enable micar's a11y fast path (--use-a11y true): accessibility tree
    # element list + tap_path batch tool. Android only.
    self._use_a11y = use_a11y
    # Optional directory for saving per-task logs and the Midscene HTML report.
    self._output_dir = output_dir
    self._step_counter = 0
    # Task instance label (e.g. "ContactsAddContact_0"), set by suite_utils
    # before each step; used to name the saved Midscene HTML report so it matches
    # the checkpoint instance.
    self._task_name: str | None = None

  def set_task_name(self, task_name: str) -> None:
    """Sets the current task instance label (used to name the HTML report)."""
    self._task_name = task_name

  def _build_cmd(self, goal: str) -> list[str]:
    cmd = self._base_cmd + [
        '--aw-goal',
        goal,
        '--device',
        self._adb_serial,
        '--platform',
        'android',
    ]
    if self._model:
      cmd += ['--model', self._model]
    if self._max_turns:
      cmd += ['--max-turns', str(self._max_turns)]
    if self._task_name:
      cmd += ['--task-name', self._task_name]
    if self._use_a11y:
      cmd += ['--use-a11y', 'true']
    return cmd

  def _hide_automation_ui(self) -> None:
    """Disables the developer-option overlays that pollute screenshots.

    android_env enables both `show_touches` (tap/slide white circles) and
    `show_pointer_location` (blue crosshair + coordinate readout) by default.
    AndroidWorld's native agents only turn off `pointer_location` (via
    `hide_automation_ui`), and the micar agent never called it, so both
    overlays showed up in micar's Midscene screenshots. Turn both off before
    running the subprocess.
    """
    # pointer_location: coordinate crosshair + top readout.
    self.env.hide_automation_ui()
    # show_touches: tap/slide trajectory circles (not covered by the above).
    adb_utils.issue_generic_request(
        'shell settings put system show_touches 0', self.env.controller
    )

  def _extract_finish_summary(self, output: str) -> str | None:
    """Extracts the `summary` from micar's `finish(summary=...)` log line.

    micar signals completion by invoking its `finish` stop tool, which the
    loop_kernel logs as a single line: `Tool: finish({"summary": "..."})`.
    The JSON args are always on one line (json.dumps without indent), so the
    substring after `finish(` up to the closing `)` is valid JSON.

    Returns the summary string, or None if no non-empty summary was found.
    """
    for line in output.splitlines():
      idx = line.find('Tool: finish(')
      if idx == -1:
        continue
      json_str = line[idx + len('Tool: finish('):].rstrip()
      # The log line ends with the closing ')' of the finish(...) call.
      if json_str.endswith(')'):
        json_str = json_str[:-1]
      try:
        args = json.loads(json_str)
      except ValueError:
        continue
      if isinstance(args, dict) and args.get('summary'):
        return args['summary']
    return None

  _ANSWER_MARKER = 'ANDROIDWORLD_ANSWER:'

  def _extract_answer(self, output: str) -> str | None:
    """Extracts the answer from micar's ``ANDROIDWORLD_ANSWER:`` marker line.

    The micar subprocess prints ``ANDROIDWORLD_ANSWER: <答案>`` when the agent
    invokes its dedicated ``answer`` stop tool (mirroring MobileWorld's
    ``MOBILEWORLD_ANSWER``). Prefer this over the finish-summary fallback: it
    carries the bare answer in the exact format the information-retrieval
    verifier expects.
    """
    for line in output.splitlines():
      if self._ANSWER_MARKER in line:
        value = line.split(self._ANSWER_MARKER, 1)[1].strip()
        if value:
          return value
    return None

  def _set_answer(self, answer: str) -> None:
    """Routes the micar agent's final answer into the environment's cache.

    Information-retrieval tasks score via ``env.interaction_cache``, which the
    framework only populates when the agent issues an ANSWER action. micar
    reports its answer through its ``answer`` stop tool (falling back to the
    ``finish(summary=...)`` tool), so we transfer that value here. Direct
    assignment avoids the ``display_message`` broadcast side effect of
    ``execute_action(ANSWER)``, which is unnecessary now that the subprocess
    has finished.
    """
    self.env.interaction_cache = answer

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    self._step_counter += 1
    self._hide_automation_ui()
    cmd = self._build_cmd(goal)

    # Inject THINKING_MODE so the eval script can control it via a flag rather
    # than requiring a shell-level `export THINKING_MODE=...`.
    env = os.environ.copy()
    if self._thinking_mode:
      env['THINKING_MODE'] = self._thinking_mode

    # No output dir: capture micar's logs and replay them to this process's
    # stdout (so progress stays visible), then extract the finish summary.
    if not self._output_dir:
      proc = subprocess.run(
          cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
      )
      if proc.stdout:
        sys.stdout.buffer.write(proc.stdout)
        sys.stdout.buffer.flush()
        text = proc.stdout.decode('utf-8', errors='replace')
        answer = self._extract_answer(text)
        if answer is None:
          answer = self._extract_finish_summary(text)
        if answer:
          self._set_answer(answer)
      return base_agent.AgentInteractionResult(
          done=True, data={'exit_code': proc.returncode},
      )

    # Output dir set: tee the combined stdout/stderr to a per-task log file and
    # to this process's stdout (so progress stays visible), and point micar's
    # Midscene HTML report at the same directory.
    os.makedirs(self._output_dir, exist_ok=True)
    env['MIDSCENE_REPORT_OUTPUT_DIR'] = self._output_dir

    if self._task_name:
      # Prefer the task instance label (e.g. "ExpenseAddMultipleFromMarkor_0"):
      # it already carries the real instance number, so re-running a batch (with
      # checkpoint resume) reuses the same filename instead of restarting a
      # per-process counter from 1.
      safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', self._task_name)
      log_path = os.path.join(self._output_dir, f'{safe_name}.log')
    else:
      safe_goal = re.sub(r'[^A-Za-z0-9_.-]+', '_', goal)[:60] or 'task'
      log_path = os.path.join(
          self._output_dir, f'{self._step_counter:03d}_{safe_goal}.log'
      )

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
    )
    output = bytearray()
    with open(log_path, 'wb') as log_file:
      assert proc.stdout is not None
      while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
          break
        output += chunk
        log_file.write(chunk)
        log_file.flush()
        sys.stdout.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    returncode = proc.wait()

    text = output.decode('utf-8', errors='replace')
    answer = self._extract_answer(text)
    if answer is None:
      answer = self._extract_finish_summary(text)
    if answer:
      self._set_answer(answer)

    return base_agent.AgentInteractionResult(
        done=True,
        data={'exit_code': returncode, 'log_path': log_path},
    )
