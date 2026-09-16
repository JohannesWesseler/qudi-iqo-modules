# HRT Battery Tester Control and Qudi Integration Plan

Status: implementation in progress; native passive monitoring, Qudi data storage, and dummy-backed orchestration implemented  
Last updated: 2026-07-23

## 1. Purpose

This document describes how to integrate the Battery Dynamics HRT battery tester into Qudi when
the tester exposes no supported programming API and is controlled through its web interface.
Qudi uses the structured REST and Socket.IO services discovered behind that interface for passive
monitoring. Browser automation is reserved for UI-only workflows that cannot be implemented
reliably through those services. Qudi provides the hardware abstraction, measurement
orchestration, state reporting, data management, and optional GUI.

The currently known tester address is:

```text
http://10.1.24.116:1841/
```

The address must remain a Qudi configuration option rather than being hard-coded. All dependency
installation and execution must use the project's working interpreter at
`C:\Users\aj92uwef\PycharmProjects\qudi-core\venv\Scripts\python.exe`.

## 2. Design summary

The tester already contains the real-time measurement and protocol engine. Qudi should not
reimplement charge/discharge steps, loops, end conditions, or recording intervals. Instead, Qudi
will:

1. upload or select a protocol;
2. create a task with battery metadata and explicit safety limits;
3. verify the configured task before starting it;
4. start, monitor, pause, resume, or abort the task;
5. reconcile state after a browser, network, or Qudi restart; and
6. download the original measurement data and store it with Qudi provenance.

The intended architecture is:

```text
Qudi GUI / notebook / ModuleTask
                |
       BatteryTesterLogic
   orchestration, polling, safety,
   ownership, signals, data storage
                |
      BatteryTesterInterface
                |
       HrtBatteryTester hardware
          /             \
 read-only REST     native Socket.IO
 queue/identity      live telemetry
          \             /
             tester engine

Future UI-only action: isolated Playwright subprocess -> HRT web UI
```

The core integration is a hardware module and a logic module. A GUI and `ModuleTask` wrappers are
optional clients of the logic module, not substitutes for it.

### 2.1 Discovery update (2026-07-22)

Read-only inspection of the live device changed one important architectural assumption. The vendor
frontend is an ExtJS application that uses structured internal services:

- frontend version `3.1.8` at port 1841;
- REST API version `1.3.6` at port 8000;
- Socket.IO at port 3000;
- protocol format configured by the frontend as version `2.2`; and
- explicit channel records for channels 13, 14, 15, and 16.

The REST API exposes system identity, channel/task queues, experiments, protocols, and data/export
resources. Socket.IO supplies live channel data and performs control actions. These services are
unsupported internal interfaces rather than a vendor-guaranteed public API, so they remain hidden
behind the HRT transport.

The implementation uses REST responses for structured read operations and a native Socket.IO
client for live state. The client implements the same subscription used by frontend 3.1.8:
websocket-only Engine.IO 3 transport, `start` on connect/reconnect, `new_system_data` receive, and
`stop` on disconnect. Browser actions remain an isolated fallback for workflows that cannot be
performed reliably through the structured services.

The system endpoint currently reports `channelCount: 5` while the channel endpoint returns four
explicit records. The implementation logs this inconsistency and treats the explicit channel IDs as
authoritative.

The implemented commissioning slice now includes the hardware-neutral models and interface, the
read-only REST client, native Socket.IO live-data transport, the HRT hardware module, a polling and
data-storage logic module, configuration, and parser/transport tests. Mutating methods on the live
HRT adapter deliberately fail with `BatteryTesterControlUnavailableError`. Live task creation and
control will remain disabled until a dedicated commissioning channel and electrically safe test
object have been identified.

### 2.2 Orchestration update (2026-07-23)

The second implementation slice adds functionality that can be exercised without the physical
tester:

- canonical protocol JSON parsing and SHA-256 identity for protocol versions 2.0, 2.1, and 2.2;
- recursive structural checks for protocol globals and nested step IDs;
- immutable task, global-override, task-reference, and experiment-reference types;
- an explicit safety envelope and live preflight checks for channel state, cell voltage,
  temperature, and controller ownership;
- reserved `qudi-run:<UUID>` ownership markers in tester comments;
- a multichannel dummy tester with idempotent queues and simulated start, charge, discharge,
  pause, resume, stop, and completion transitions; and
- guarded logic methods and signals for submission and execution.

The two hardware implementations intentionally advertise different capabilities. The dummy sets
`task_control_available=True`, allowing full orchestration tests. `HrtBatteryTester` continues to
set it to `False`, so logic-level control calls fail before any live browser or device mutation.
Use `config_examples/battery_tester_dummy.cfg` for safe orchestration development.

### 2.3 Passive-monitoring update (2026-07-23)

The passive monitoring stage is operational against the installed tester:

- channels 13, 14, 15, and 16 are discovered through REST and receive fresh Socket.IO telemetry;
- voltage, current, auxiliary/terminal voltage, temperature, state, queue, experiment, and protocol
  fields are normalized into immutable `ChannelSnapshot` values;
- idle controller uptime and step counters are intentionally cleared because they are not
  experiment runtime;
- `BatteryTesterLogic` can create a monitoring session and append one long-format row per channel
  per poll through Qudi `TextDataStorage`;
- row-count and time-based flushing, bounded buffering, connection-error rows, metadata, and a
  final deactivation flush are implemented; and
- the headless example can auto-activate hardware and logic, auto-start monitoring, and create a
  voltage/current PNG beside the data file.

Plot rendering runs in a short-lived non-interactive subprocess. This avoids loading a
Matplotlib/Qt figure into the long-running headless Qudi process and keeps plotting failures
isolated from acquisition and final data flushing.

An actual Windows headless Qudi run successfully activated the native transport, saved data, and
shut down cleanly. The earlier synchronous Playwright worker repeatedly caused a native access
violation during Python process exit even after orderly browser shutdown. It is therefore not an
acceptable in-process production telemetry transport on this deployment. If a future control
workflow truly requires browser automation, it must be placed in a separately supervised process
with plain-data IPC so a browser failure cannot take down Qudi.

## 3. Information established from the supplied documentation

The manuals and protocol examples in this directory establish the following relevant behavior:

- The device UI is served over Ethernet on port 1841.
- The device can have multiple independent channels. The exact installed channel count must be
  discovered at runtime instead of inferred from a manual.
- A task contains a test name, protocol, battery model, battery number, safety limits, comments,
  and potentially advanced settings or global-value overrides.
- Protocols are JSON documents containing globals, steps, loops, end conditions, recording
  intervals, and optional formulas or expert settings.
- A running task can be paused and resumed using the play/pause control. A separate Stop control
  aborts the running task without removing its queue entry.
- The device maintains an experiment database and can export measurement data as CSV.
- The device can continue executing its protocol independently of the browser and Qudi.
- Multiple users editing the same task can conflict. Qudi must therefore avoid concurrent task
  editing with a human operator.
- A missing temperature sensor is represented by approximately `-99 degC`, but the tester may
  still allow the task to start. Qudi must reject this by default.
- The external SAFETY connector is an independent hardware interlock. Web automation is not an
  emergency-stop or safety system.

Important device status codes include:

| Device state | Meaning |
| --- | --- |
| `0` | Standby, no protocol running |
| `1`, `7`, `8`, `51`, `52` | Internal/transitional state |
| `2` | Legacy pause state |
| `12`-`19` | Legacy charging state |
| `20`-`29` | Legacy discharging state |
| `40` | Starting a protocol |
| `41`, `80`-`89` | Paused protocol |
| `99` | Error |
| `100`-`159` | Charging state |
| `200`-`259` | Discharging state |

Important data event codes include:

| Event | Meaning |
| --- | --- |
| `30` | Normal end of protocol |
| `100`-`109` | Global safety-limit violation |
| `140` | Sense and power voltage mismatch |
| `150` | Error raised by a step end condition |
| `190` | Stop command received |
| `191` | Pause command received |
| `192` | Resume command received |
| `193` | Manual channel navigation command |
| `199` | General error; inspect the additional info/var fields |

These codes should be mapped to typed states and termination reasons rather than exposed as
uninterpreted integers to users of the Qudi logic API.

## 4. Proposed source layout

```text
qudi-iqo-modules/src/qudi/
|-- interface/
|   |-- battery_tester_interface.py
|   `-- battery_tester_models.py
|-- hardware/
|   |-- battery_tester/
|   |   |-- __init__.py
|   |   |-- hrt_battery_tester.py
|   |   |-- hrt_rest_client.py
|   |   |-- socketio_transport.py
|   |   |-- playwright_transport.py
|   |   |-- selectors/
|   |   |   `-- hrt_<supported-version>.py
|   |   `-- protocols/
|   `-- dummy/
|       `-- battery_tester_dummy.py
|-- logic/
|   `-- battery_tester_logic.py
`-- gui/
    `-- battery_tester/                 # later phase
```

Additional changes will include:

- a configuration example under `qudi-iqo-modules/config_examples/`;
- a version-pinned native Socket.IO optional dependency and a separate Playwright fallback extra;
- package-data rules for protocol JSON files if bundled templates are intended to be installed;
- parser and browser fixture tests; and
- hardware-in-the-loop test procedures that are never part of the normal automated test suite.

The PDF manuals are reference documentation and do not need to become runtime package data.

## 5. Hardware-neutral interface

`BatteryTesterInterface` should describe battery-test operations, not clicks or web pages. The HRT
implementation will inherit this interface in the same way as other Qudi hardware modules.

Initial interface operations should be similar to:

```python
@property
def capabilities(self) -> BatteryTesterCapabilities: ...

def get_channel_snapshots(self) -> tuple[ChannelSnapshot, ...]: ...

def submit_task(self, spec: BatteryTaskSpec) -> TaskReference: ...

def start_task(self, task: TaskReference) -> ExperimentReference: ...

def pause_task(self, channel: int) -> None: ...

def resume_task(self, channel: int) -> None: ...

def stop_task(self, channel: int) -> None: ...

def get_experiment(self, experiment_id: str) -> ExperimentSnapshot: ...

def download_experiment(self, experiment_id: str, destination: str) -> str: ...
```

The concrete signatures may change during implementation, but the following rules should hold:

- returned objects are immutable snapshots or references;
- methods have explicit timeouts and raise structured exceptions;
- channel numbers and experiment identifiers are never inferred from visual position alone;
- mutating operations verify a postcondition before reporting success; and
- the interface never exposes `Page`, `Locator`, browser selectors, or other Playwright objects.

### 5.1 Core data types

`BatteryTaskSpec` should contain at least:

- a Qudi-generated run UUID;
- test name;
- protocol JSON bytes or an immutable protocol path;
- protocol SHA-256 checksum;
- battery model and battery number/serial;
- explicit voltage, current, and temperature safety limits;
- optional protocol-global overrides;
- operator comments and other provenance; and
- requested channel.

Other useful types are:

- `BatteryTesterCapabilities`;
- `SafetyLimits`;
- `ChannelSnapshot`;
- `ChannelOperatingState` enum;
- `TaskReference`;
- `ExperimentReference` and `ExperimentSnapshot`; and
- `TerminationReason` enum.

## 6. HRT web-service and hardware implementation

### 6.1 Transport ownership and threading

`HrtBatteryTester` owns two read-only transports. The REST client performs bounded request/response
operations for identity, channels, and queues. `SocketIoHrtTransport` owns a Socket.IO client and
caches plain Python copies of the most recent `new_system_data` event under a lock. Qudi polling
only reads that cache; it never manipulates Socket.IO objects directly.

The live client emits `start` after every initial connection or automatic reconnect and emits
`stop` before disconnecting. These event names subscribe and unsubscribe the client from telemetry;
they do not start or stop a tester experiment. Cached channel data has a configured maximum age so
a connected-but-stalled feed is reported as disconnected rather than silently returning old data.

The normal Chrome tab remains useful for manual exploration. It is not required for production
monitoring and Qudi does not attach to the operator's profile.

The in-process synchronous Playwright worker is retained only as an explicit diagnostic fallback.
It is not safe for unattended use on the commissioned Windows system because it reproduced a
native access violation at process exit. Future UI-only automation must run in a child process that
owns its browser for its whole lifetime, receives serialized commands, and returns only plain data.
Qudi must remain able to supervise, time out, and replace that process.

### 6.2 Configuration options

Expected hardware `ConfigOption` values include:

```yaml
hardware:
    hrt_battery_tester:
        module.Class: 'battery_tester.hrt_battery_tester.HrtBatteryTester'
        options:
            base_url: 'http://10.1.24.116:1841/'
            api_url: 'http://10.1.24.116:8000/'
            live_transport: 'socketio'
            socketio_url: 'http://10.1.24.116:3000/'
            socketio_path: 'socket.io'
            max_live_age_s: 5
            request_timeout_s: 10
            reconnect_attempts: 2
            reconnect_backoff_s: 0.5
            reconnect_cooldown_s: 10
            allowed_channels: [13, 14, 15, 16]
            require_live_data: true
```

The Socket.IO dependency versions are deliberately bounded to the legacy protocol used by
frontend 3.1.8: `python-socketio` 4.x and `python-engineio` 3.x. The hardware module refuses an
unvalidated frontend version rather than assuming wire compatibility after a device update.

### 6.3 Versioned page adapter

All UI knowledge should be isolated behind a small page-adapter layer. On activation, the driver
must read the displayed HRT software version and select a supported selector profile.

Selector priority is:

1. stable element IDs or `data-*` attributes;
2. accessible roles and labels;
3. stable visible text; and
4. tightly scoped structural selectors as a last resort.

Screen coordinates, arbitrary click positions, fixed sleeps, and broad positional selectors such
as `nth(3)` are not acceptable for control actions.

If the detected software version is unknown, the driver should fail safe: diagnostic/read-only
operations may remain available if they can be proven reliable, but task creation and control must
be refused until the adapter is validated.

### 6.4 Network inspection

Although there is no supported public API, the web application communicates with an internal
backend. The initial exploration phase must capture HTTP and WebSocket traffic while a human
performs the supported workflow.

The discovered REST resources and Socket.IO events are now the primary passive-monitoring
transport. They remain isolated inside the HRT hardware package because the endpoints are
unsupported and may change with software updates. Frontend/API versions are recorded in every
monitoring file, and the public Qudi interface remains unchanged if a version-specific transport
must be replaced.

### 6.5 Diagnostics

On an unexpected page or failed postcondition, the transport should save:

- a screenshot;
- a DOM or accessibility snapshot;
- the detected software version and current URL;
- recent browser console errors; and
- a Playwright trace for mutating workflows when tracing is enabled.

Diagnostics must be written to a bounded directory and use unique timestamps/run IDs. Routine
polling must not generate screenshots or traces.

## 7. Command reliability and idempotency

Browser actions can have ambiguous outcomes: a click may reach the tester even if the subsequent
wait times out. Mutating commands must not be retried blindly.

Each task submission, start, pause, resume, and stop operation must use this sequence:

1. read and validate the current channel/task state;
2. confirm that the expected task or experiment is selected;
3. perform exactly one action;
4. wait for a precise state transition or identifier;
5. on timeout, reread and reconcile the actual device state; and
6. retry only when it is certain the original action did not occur.

The Qudi run UUID should be included in the test name or comment. This makes task submission
idempotent and allows Qudi to recover ownership after a restart. The protocol checksum and battery
identity provide additional comparison fields.

Qudi must not silently modify or stop a test that lacks its ownership marker. Externally started
tests should be reported as observed/unmanaged.

## 8. Logic module responsibilities

`BatteryTesterLogic` is the stable API for GUIs, notebooks, remote modules, and higher-level
experiment orchestration. It should provide:

- periodic state polling with a `QTimer` in the logic thread;
- per-channel snapshots and change signals;
- guarded task submission and start methods;
- pause, resume, and explicit abort methods;
- ownership tracking by run UUID and tester experiment ID;
- connection-loss and browser-restart reconciliation;
- completion classification using state and data event codes;
- data export and provenance storage; and
- optional coordination hooks for other Qudi logic modules.

Example signals include:

```python
sigConnectionStateChanged = QtCore.Signal(bool, str)
sigChannelStateChanged = QtCore.Signal(int, object)
sigTaskSubmitted = QtCore.Signal(object)
sigExperimentStarted = QtCore.Signal(object)
sigExperimentFinished = QtCore.Signal(object)
sigDataExported = QtCore.Signal(str, str)
sigError = QtCore.Signal(str)
```

The logic module should use its Qudi `module_state` as a coarse aggregate: it is locked while at
least one Qudi-owned channel is running or paused. Detailed state remains per channel.

Status polling should initially occur every 1-2 seconds and be configurable. It reads a locked
copy of the Socket.IO cache and combines it with REST queue data. Future mutating commands must be
serialized independently so they cannot overlap with another mutation.

## 9. Safety policy

Browser automation and Qudi are supervisory software, not safety mechanisms. Device safety limits,
correct cell wiring, the temperature sensor, and the external SAFETY connector remain mandatory.

The first implementation must enforce these rules:

- Every start requires explicit per-run safety limits.
- Safety values are read back from the final task form and compared before clicking Run/Start.
- Configured laboratory hard ceilings are applied in addition to cell-specific limits.
- A temperature near `-99 degC` is treated as a missing sensor and blocks start by default.
- The selected channel must be idle and must not contain an unidentified active task.
- The submitted protocol is checksummed and preserved exactly.
- The tester's protocol Review function is run before start where practical.
- Review errors block start. Warnings require an explicit override recorded in provenance.
- Parallel-channel operation is out of scope for the MVP and disabled by default.
- A connection loss never implies that the autonomous test stopped.
- Reconnection never triggers an automatic start, resume, or stop.
- Hardware deactivation closes browser resources but does not abort active tests.
- The browser Stop action is called `abort` or `stop_task`, never `emergency_stop`.

Protocol validation in Qudi should check JSON syntax, supported versions, structure, known units,
and obviously invalid values. It should not attempt to duplicate the complete device protocol
engine, especially formula evaluation and sign conversion. Device safety limits provide the
independent execution envelope.

## 10. Protocol handling

For the MVP, protocols should be loaded from JSON files and uploaded directly into the task dialog.
Qudi should avoid modifying the shared protocol database on the tester. This reduces conflicts and
makes each run reproducible.

For every submitted protocol:

1. read bytes once;
2. parse and perform basic validation;
3. calculate a SHA-256 checksum;
4. upload those exact bytes;
5. record the checksum with the task; and
6. copy those exact bytes beside the exported measurement data.

The supplied templates include protocol versions 2.0 and 2.1 and use more than one step-name
convention. The loader must preserve these files rather than normalizing or rewriting them during
the first implementation.

Protocol authoring or a Qudi protocol editor is explicitly out of scope until upload, execution,
and data recovery are robust.

## 11. Measurement data and provenance

The tester remains the authoritative source of measurement samples. Qudi should not repeatedly
download a complete CSV merely to provide live monitoring.

On completion or explicit user request, the logic module should:

1. select the experiment by its captured tester ID;
2. trigger a single-experiment CSV download;
3. wait for the Playwright download to complete in a temporary location;
4. move it into `BatteryTesterLogic.module_default_data_dir` using a Qudi-style timestamped run
   directory or filename;
5. store the exact submitted protocol beside it; and
6. store a metadata/provenance file.

Required provenance includes:

- Qudi run UUID;
- tester experiment and task IDs;
- device address and reported device/software version;
- channel or parallel-channel group;
- test name, battery model, and battery number;
- exact safety limits and global overrides;
- protocol filename, version, and checksum;
- start/end timestamps;
- final channel state, device event code, and normalized termination reason; and
- Qudi/browser errors or reconnection events that occurred during the run.

The original vendor CSV must be preserved verbatim. It contains variable-length metadata sections
and a final `[DATA]` section, so passing it directly through Qudi's `CsvDataStorage` would alter its
representation. A separate parser can produce a normalized Qudi `.dat` or NumPy derivative when
needed for plotting and cross-module analysis.

The parser must recognize at least the documented format evolution:

- SI units in newer exports;
- `Time_h` versus `RunTime_s`;
- `info64`/`var64` versus `info`/`var`; and
- optional `calc1` through `calc4` columns.

`StatusVar` is suitable for small user preferences and a compact ownership map. It should be
manually dumped only at important ownership transitions, not on every poll. Recovery must also be
possible by finding the Qudi run UUID in the tester's task/experiment metadata because an abrupt
process termination may bypass status-variable persistence.

## 12. GUI and orchestration API

The initial implementation must be fully controllable through the logic module without a GUI. A
later GUI should remain thin and contain no measurement logic.

A useful GUI would provide:

- connection and detected-version status;
- one card/table row per channel;
- current state, task, step, voltage, current, temperature, and elapsed time when reliably
  available;
- protocol and battery selection;
- explicit safety-limit review;
- separate Start, Pause, Resume, and Abort controls;
- clear distinction between Qudi-owned and externally owned tests;
- completion/error history; and
- export status and a link/path to stored data.

`TaskRunnerLogic` or custom `ModuleTask` classes may later orchestrate a battery test together with
other Qudi instruments. They should call `BatteryTesterLogic` and wait on its experiment signals.
They must not directly access the hardware module or a concrete transport.

## 13. Implementation phases

### Phase 0: Safe exploration

- Use the currently reachable `http://10.1.24.116:1841/` UI.
- Record the exact displayed HRT software version and installed channels.
- Use Playwright code generation/tracing or a small standalone script outside Qudi.
- Capture the System, task dialog, queue, running-state, experiment, raw-data, and export flows.
- Inspect network requests and WebSocket messages.
- Perform read-only operations first.
- Identify a dedicated commissioning channel and safe test object before any automated start.

Deliverable: a selector/network inventory and a standalone read-only proof of concept.

### Phase 1: Interface and read-only native transports

- Define data models, enums, exceptions, and `BatteryTesterInterface`.
- Implement REST identity/queue access and native Socket.IO live telemetry.
- Add version detection and fail-safe compatibility gating.
- Implement connect, capability discovery, and channel snapshots.
- Add freshness checks and bounded reconnection.

Deliverable: a Qudi hardware module capable of reliable read-only operation.

Implementation status: complete for frontend 3.1.8/API 1.3.6 and channels 13-16. A real headless
Qudi run verified activation, fresh telemetry, data storage, and clean process shutdown.

### Phase 2: Dummy hardware and logic

- Implement `BatteryTesterDummy` with realistic multichannel transitions.
- Implement logic polling, signals, ownership, and aggregate module state.
- Implement passive monitoring sessions using Qudi `TextDataStorage`.
- Add protocol loading/checksumming and safety validation without browser mutation.
- Add a configuration example.

Deliverable: orchestration can be developed and tested without the real tester.

Implementation status: the dummy, polling/control logic, protocol identity, ownership markers,
task safety envelope, preflight checks, and example configuration are implemented. Persisted
ownership reconciliation and a richer semantic protocol linter remain later hardening work.

### Phase 3: Guarded task submission

- Upload a protocol into a selected channel task dialog.
- Fill metadata, global overrides, and safety limits.
- Save to the queue without starting.
- Reopen/read back all critical values and compare to the requested specification.
- Detect duplicate run UUIDs and operator conflicts.

Deliverable: idempotent queue submission with no automatic electrical activity.

### Phase 4: Controlled execution

- Enable Start on one independent channel.
- Capture the resulting task and experiment identifiers.
- Implement and verify pause, resume, and explicit abort.
- Map channel and event codes to normalized outcomes.
- Reconcile state after page reload, browser restart, network interruption, and Qudi restart.

Deliverable: safe single-channel execution and monitoring.

### Phase 5: Export and analysis integration

- Download original CSV data by experiment ID.
- Save raw data, protocol, and provenance atomically in the Qudi data directory.
- Implement the version-tolerant CSV parser.
- Optionally create normalized Qudi data and plots.

Deliverable: complete reproducible measurement records.

### Phase 6: GUI and multichannel support

- Add a thin Qudi GUI.
- Validate several independent channels running concurrently.
- Add higher-level orchestration hooks or `ModuleTask` wrappers.
- Treat parallel-channel operation as a separately reviewed feature.

Deliverable: routine laboratory operation from Qudi.

## 14. Testing strategy

### 14.1 Pure unit tests

- Protocol JSON parsing, versions, checksums, and validation.
- Safety-limit validation and temperature-sensor detection.
- State-code and event-code mappings.
- Task identity and idempotency logic.
- Old/new CSV headers, units, metadata sections, and malformed exports.
- State reconciliation after restart.

### 14.2 Transport and browser-fallback tests

- Socket event normalization, cache freshness, reconnect, and malformed payload tests.
- Sanitized HTML/accessibility fixtures for selector tests.
- A small local mock web application for the required dynamic workflows.
- Download handling, timeouts, diagnostics, and unknown-version behavior.
- Tests proving that a timed-out mutation is reconciled rather than blindly repeated.

### 14.3 Dummy integration tests

- Full logic flow from submission to completion.
- Multiple independent channel states.
- Pause/resume/abort transitions.
- Safety failures and export failures.
- Qudi module activation and deactivation behavior.

### 14.4 Hardware-in-the-loop commissioning

Hardware-in-the-loop tests require an approved test setup and human supervision. The sequence is:

1. read-only capability and experiment access;
2. export of an existing historical experiment;
3. creation and deletion of a queued task without starting it;
4. a deliberately low-energy, short protocol on one channel;
5. pause and resume;
6. explicit abort;
7. network cable interruption while the tester continues autonomously;
8. browser crash/restart;
9. Qudi restart and ownership reconciliation; and
10. controlled concurrent human access to confirm conflict detection.

Parallel-channel operation is excluded from this commissioning sequence.

## 15. MVP acceptance criteria

The first useful release is complete when it can demonstrate all of the following:

- It uses the project's `venv` and provides reproducible native Socket.IO dependencies, with a
  separately installable Playwright fallback.
- Qudi activation discovers the device software version and actual channel set.
- Existing running tests are visible without being adopted or modified.
- An immutable protocol can be queued and read back without starting.
- Missing sensor, unsafe limits, unknown software versions, and ownership conflicts block start.
- A test starts exactly once and its experiment ID is captured.
- Running, paused, completed, aborted, disconnected, and error states are distinguished.
- Browser or network failure cannot cause duplicate submission or duplicate start.
- Qudi reconnects and reconciles a test that continued autonomously.
- Deactivating Qudi does not unintentionally stop a test.
- The original CSV and protocol are saved with complete provenance.
- The entire workflow is available through the logic API without requiring a GUI.

## 16. Decisions and information still needed

The exploration and commissioning phases must settle these points before production use:

- exact HRT software version currently served by the tester;
- installed channel count and channel naming;
- whether authentication/session expiry exists;
- exact DOM selectors and internal network messages;
- which live quantities can be read without altering global UI settings;
- the complete set and units of safety-limit fields;
- exact task and experiment identifiers exposed by the UI/backend;
- desired policy for users manually operating the web UI while Qudi is connected;
- designated commissioning channel and safe reference cell/test setup; and
- whether normalized Qudi data is required immediately or raw vendor CSV is sufficient for the
  first release.

The default answers for the MVP are: one independent commissioning channel, exclusive Qudi task
editing, raw CSV preservation, no parallel operation, and no Qudi protocol editor.
