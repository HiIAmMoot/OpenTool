import ctypes
import os
from pathlib import Path
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from lightmapper import blender_find, gpu
from lightmapper.memory import free_physical_bytes
from lightmapper.progress import STAGE_LABELS, fraction
from . import devices, errors, estimate as estimate_module, form, jobobject, presets, runner, usersettings
from .estimate import EstimateInputs


class _BakeGate:
    # held by the bake thread's emit wrapper and by on_close, so "is a tag write under way" is
    # decided in one place instead of from the UI thread's slightly stale view of the events.
    def __init__(self):
        self.lock = threading.Lock()
        self.closing = False
        self.writing = False


class App:
    def __init__(self, root: tk.Misc, *, store_path: Path | None = None,
                 find_blender=blender_find.find_blender, run_bake=runner.run_bake,
                 log_path: Path | None = None, probe_gpu=gpu.probe_adapters,
                 probe_denoiser=devices.probe_denoiser,
                 estimate=estimate_module.run_estimate):
        self.root = root
        self.window = root.winfo_toplevel()
        self.tk_root = root._root()
        self.store_path = store_path
        self.log_path = Path(log_path) if log_path is not None else Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'OpenTool' / 'opentool.log'
        self.find_blender = find_blender
        self._run_bake = run_bake
        self._probe_gpu = probe_gpu
        self._probe_denoiser = probe_denoiser
        self._denoiser = None
        self._using_stage_rows = False
        self._estimate = estimate
        self.generation = 0
        self.estimate_state = 'idle'
        self._estimate_inputs: EstimateInputs | None = None
        self._estimate_running: tuple | None = None
        self._estimate_pending: tuple | None = None
        self._adapters: list | None = None
        self._gpu_reason: str | None = None
        self.busy = False
        self._live_warnings: set[str] = set()
        self.stage: str | None = None
        self.close_after_write = False
        self._locked: dict = {}
        self._gate = _BakeGate()
        self.events: queue.Queue = queue.Queue()
        self.blender: blender_find.Blender | None = None
        self._user_chose_blender = False
        self._setting_preset = False
        self._tags: Path | None = None
        self._scenario = ''
        self._scenario_file: Path | None = None
        self._bsps: list[str] = []
        self.settings = usersettings.load(store_path)

        self.window.title('OpenTool lightmapper')
        self.window.minsize(640, 560)
        self.window.resizable(True, True)
        self.tk_root.report_callback_exception = self._report_callback_exception
        self.window.protocol('WM_DELETE_WINDOW', self.on_close)
        self._build()
        self._restore()
        for value in (*self.advanced_vars.values(), self.preset_var, self.device_var, self.bsp_var,
                      self.new_names_var, self.backups_var, self.lightmap_32bit_var, self.verbose_var):
            value.trace_add('write', lambda *_args: self._reset_used_rows())
        self._set_blender_status('Looking for Blender…')
        self._discover_blender()
        self._discover_gpu()
        self._discover_denoiser()
        root.after(100, self._poll)

    def _build(self):
        self.frame = ttk.Frame(self.window, padding=12)
        self.frame.pack(fill='both', expand=True)
        self.frame.columnconfigure(0, weight=1)

        tags_row = ttk.Frame(self.frame)
        tags_row.grid(row=0, column=0, sticky='ew')
        tags_row.columnconfigure(0, weight=1)
        self.tags_label = ttk.Label(tags_row, text='HCEK tags folder')
        self.tags_label.grid(row=0, column=0, sticky='w')
        self.tags_entry = ttk.Entry(tags_row)
        self.tags_entry.grid(row=1, column=0, sticky='ew', pady=(2, 0))
        ttk.Button(tags_row, text='Browse…', command=self._browse_tags).grid(row=1, column=1, padx=(6, 0))
        self.tags_error = ttk.Label(self.frame, foreground='red')
        self.tags_error.grid(row=1, column=0, sticky='w')
        self.tags_error.grid_remove()
        self.tags_entry.bind('<FocusOut>', self._commit_tags)
        self.tags_entry.bind('<Return>', self._commit_tags)

        scenario_row = ttk.Frame(self.frame)
        scenario_row.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        scenario_row.columnconfigure(0, weight=1)
        ttk.Label(scenario_row, text='Scenario').grid(row=0, column=0, sticky='w')
        self.scenario_var = tk.StringVar()
        self.scenario_entry = ttk.Entry(scenario_row, textvariable=self.scenario_var, state='readonly')
        self.scenario_entry.grid(row=1, column=0, sticky='ew', pady=(2, 0))
        ttk.Button(scenario_row, text='Browse…', command=self._browse_scenario).grid(row=1, column=1, padx=(6, 0))
        self.scenario_error = ttk.Label(self.frame, foreground='red')
        self.scenario_error.grid(row=3, column=0, sticky='w')
        self.scenario_error.grid_remove()

        bsp_row = ttk.Frame(self.frame)
        bsp_row.grid(row=4, column=0, sticky='ew', pady=(8, 0))
        ttk.Label(bsp_row, text='BSP').pack(side='left')
        self.bsp_var = tk.StringVar()
        self.bsp_box = ttk.Combobox(bsp_row, textvariable=self.bsp_var, state='readonly')
        self.bsp_box.pack(side='left', fill='x', expand=True, padx=(12, 0))
        self.bsp_box.bind('<<ComboboxSelected>>', lambda _event: self._inputs_changed())
        self.bsp_message = ttk.Label(self.frame)
        self.bsp_message.grid(row=5, column=0, sticky='w')
        self.bsp_message.grid_remove()

        quality_block = ttk.Frame(self.frame)
        quality_block.grid(row=6, column=0, sticky='ew', pady=(12, 0))
        quality = ttk.Frame(quality_block)
        quality.grid(row=0, column=0, sticky='w')
        ttk.Label(quality, text='Quality').pack(side='left')
        self.preset_var = tk.StringVar(value='Normal')
        for name in (*presets.PRESET_NAMES, presets.CUSTOM):
            ttk.Radiobutton(quality, text=name, value=name, variable=self.preset_var,
                            command=lambda selected=name: self._select_preset(selected)).pack(side='left', padx=(8, 0))
        self.preset_hint = ttk.Label(quality, text=presets.PRESET_HINTS['Normal'])
        self.preset_hint.pack(side='left', padx=(12, 0))
        self.preset_note = ttk.Label(quality_block)
        self.preset_note.grid(row=1, column=0, sticky='w')

        self.new_names_var = tk.BooleanVar(value=self.settings.new_names)
        options = ttk.Frame(self.frame)
        options.grid(row=7, column=0, sticky='w', pady=(8, 0))
        self.new_names_check = ttk.Checkbutton(
            options, text='Write to new tag names', variable=self.new_names_var,
            command=self._update_new_names_hint,
        )
        self.new_names_check.grid(row=0, column=0, sticky='w')
        self.backups_var = tk.BooleanVar(value=self.settings.backups)
        self.backups_check = ttk.Checkbutton(
            options, text='Keep backups', variable=self.backups_var,
            command=self._update_new_names_hint,
        )
        self.backups_check.grid(row=0, column=1, sticky='w', padx=(12, 0))
        self.lightmap_32bit_var = tk.BooleanVar(value=self.settings.lightmap_32bit)
        ttk.Checkbutton(options, text='32-bit lightmaps', variable=self.lightmap_32bit_var).grid(
            row=0, column=2, sticky='w', padx=(12, 0))
        self.new_names_hint = ttk.Label(self.frame)
        self.new_names_hint.grid(row=8, column=0, sticky='w')
        self._update_new_names_hint()

        self.advanced_button = ttk.Button(self.frame, text='Advanced ▸', command=self._toggle_advanced)
        self.advanced_button.grid(row=9, column=0, sticky='w', pady=(8, 0))
        self.advanced_frame = ttk.Frame(self.frame)
        self.advanced_frame.grid(row=10, column=0, sticky='ew')
        self.advanced_frame.grid_remove()
        self.advanced_frame.columnconfigure(1, weight=1)
        self.advanced_vars = {}
        self.advanced_entries = {}
        self._advanced_open = False
        for index, name in enumerate(presets.ADVANCED):
            ttk.Label(self.advanced_frame, text=presets.LABELS[name]).grid(row=index, column=0, sticky='w', padx=(0, 8), pady=2)
            value = tk.StringVar()
            self.advanced_vars[name] = value
            if name in ('bounces', 'samples', 'vertex_rays', 'sky_grid'):
                maximum = {'bounces': 8, 'samples': 4096, 'vertex_rays': 65536, 'sky_grid': 8}[name]
                entry = ttk.Spinbox(self.advanced_frame, from_=0 if name == 'bounces' else 1,
                                    to=maximum, textvariable=value)
            elif name == 'page_max':
                entry = ttk.Combobox(self.advanced_frame, values=presets.PAGE_MAX_CHOICES, textvariable=value, state='readonly')
            elif name == 'texels_per_unit':
                entry = ttk.Combobox(self.advanced_frame, values=presets.density_values(''), textvariable=value, state='readonly')
            else:
                entry = ttk.Entry(self.advanced_frame, textvariable=value)
            entry.grid(row=index, column=1, sticky='ew', pady=2)
            self.advanced_entries[name] = entry
            if name == 'page_max':
                self.page_max_hint = ttk.Label(self.advanced_frame, foreground='gray')
                self.page_max_hint.grid(row=index, column=2, sticky='w', padx=(8, 0))
            value.trace_add('write', lambda *_args, field=name: self._advanced_value_changed(field))
            if name in ('texels_per_unit', 'page_max'):
                value.trace_add('write', lambda *_args: self._inputs_changed())
        self.advanced_error = ttk.Label(self.advanced_frame, foreground='red')
        self.advanced_error.grid(row=len(presets.ADVANCED), column=0, columnspan=2, sticky='w')

        blender_row = ttk.Frame(self.frame)
        blender_row.grid(row=11, column=0, sticky='ew', pady=(12, 0))
        self.blender_label = ttk.Label(blender_row, text='Looking for Blender…')
        self.blender_label.pack(side='left', fill='x', expand=True)
        ttk.Button(blender_row, text='Change…', command=self._choose_blender).pack(side='right')

        device_row = ttk.Frame(self.frame)
        device_row.grid(row=12, column=0, sticky='ew', pady=(8, 0))
        ttk.Label(device_row, text='Device').grid(row=0, column=0, sticky='w')
        self.device_var = tk.StringVar(value=self._device_label(self.settings.device))
        self.device_box = ttk.Combobox(device_row, textvariable=self.device_var, state='readonly', width=10,
                                       values=[label for label, _ in presets.DEVICES])
        self.device_box.grid(row=0, column=1, padx=(12, 0))
        self.device_box.bind('<<ComboboxSelected>>', lambda _event: self._update_device_hint())
        self.gpu_label = ttk.Label(device_row, text=devices.gpu_text(None, None))
        self.gpu_label.grid(row=0, column=2, sticky='w', padx=(12, 0))
        self.device_hint = ttk.Label(device_row)
        self.device_hint.grid(row=1, column=1, columnspan=2, sticky='w', padx=(12, 0))
        self.stage_rows = []
        for row in range(3):
            name, state, detail = (ttk.Label(device_row) for _ in range(3))
            name.grid(row=row + 2, column=0, sticky='w', pady=(4 if row == 0 else 0, 0))
            state.grid(row=row + 2, column=1, sticky='w', padx=(12, 0), pady=(4 if row == 0 else 0, 0))
            detail.grid(row=row + 2, column=2, sticky='w', padx=(12, 0), pady=(4 if row == 0 else 0, 0))
            self.stage_rows.append((name, state, detail))
        self._show_stage_rows(devices.stage_rows(self._device_value(), None, None))

        controls = ttk.Frame(self.frame)
        controls.grid(row=13, column=0, sticky='ew', pady=(12, 0))
        self.bake_button = ttk.Button(controls, text='Bake', command=self.start_bake, state='disabled')
        self.bake_button.pack(side='left')
        self.memory_label = ttk.Label(controls, wraplength=480)
        self.memory_label.pack(side='left', padx=(12, 0))
        self._output_folder: Path | None = None

        self.stage_label = ttk.Label(self.frame, text='Ready')
        self.stage_label.grid(row=14, column=0, sticky='w', pady=(8, 0))
        self.progress = ttk.Progressbar(self.frame, maximum=1000, mode='determinate')
        self.progress.grid(row=15, column=0, sticky='ew')
        self.device_label = ttk.Label(self.frame)
        self.device_label.grid(row=16, column=0, sticky='w')
        self.device_label.grid_remove()
        self.log_box = ScrolledText(self.frame, height=7, state='disabled', wrap='word', font='TkFixedFont')
        self.log_box.grid(row=17, column=0, sticky='nsew', pady=(4, 0))
        self.frame.rowconfigure(17, weight=1)
        self.open_button = ttk.Button(self.frame, text='Open output folder', command=self._open_output, state='disabled')
        self.open_button.grid(row=18, column=0, sticky='e', pady=(4, 0))
        self.verbose_var = tk.BooleanVar(value=self.settings.verbose)
        ttk.Checkbutton(self.frame, text='Detailed log', variable=self.verbose_var).grid(row=19, column=0, sticky='w')

    def _restore(self):
        self._select_preset(self.settings.preset)
        for name, value in self.settings.advanced.items():
            if name in self.advanced_vars:
                if name == 'texels_per_unit':
                    try:
                        value = presets.format_density(float(value))
                    except (TypeError, ValueError):
                        pass
                self.advanced_vars[name].set(value)
        if self.settings.advanced_open:
            self._toggle_advanced()
        if not self.settings.tags:
            return
        try:
            tags = Path(self.settings.tags)
            if form.tags_folder_problem(tags) is not None:
                return
            self.set_tags(self.settings.tags)
            scenario_file = self._scenario_file_for(self.settings.scenario)
            if scenario_file is not None and scenario_file.is_file():
                try:
                    self.set_scenario_file(scenario_file)
                    if self.scenario_error['text']:
                        raise ValueError(self.scenario_error['text'])
                except Exception as exc:
                    self._clear_scenario()
                    self._write_exception(exc)
                if self.settings.bsp in self._bsps:
                    self.bsp_var.set(self.settings.bsp)
                    self._inputs_changed()
        except Exception as exc:
            self.set_tags('')
            self._clear_scenario()
            self._write_exception(exc)

    def _scenario_file_for(self, scenario):
        if not scenario or self._tags is None:
            return None
        return self._tags / (scenario.replace('\\', os.sep) + '.scenario')

    def set_tags(self, path: str):
        new_tags = Path(path) if path else None
        changed = new_tags != self._tags
        self.tags_entry.delete(0, 'end')
        self.tags_entry.insert(0, path)
        self._tags = new_tags
        problem = form.tags_folder_problem(new_tags) if new_tags is not None else None
        self._set_optional_label(self.tags_error, problem or '')
        if problem or changed:
            self._clear_scenario()
        self._inputs_changed()

    def _commit_tags(self, _event=None):
        self.set_tags(self.tags_entry.get())
        return 'break'

    def set_scenario_file(self, path: Path):
        self._scenario_file = Path(path)
        self.scenario_var.set(str(self._scenario_file))
        try:
            self._scenario = form.scenario_tag_path(self._tags, self._scenario_file)
            self._bsps = form.list_bsps(self._tags, self._scenario)
            self._set_optional_label(self.scenario_error, '')
            self._set_optional_label(
                self.bsp_message,
                '' if self._bsps else 'This scenario has no BSP, so there is nothing to light.',
            )
            self.bsp_box.configure(values=self._bsps)
            self.bsp_var.set(self._bsps[0] if len(self._bsps) == 1 else '')
        except Exception as exc:
            self._scenario = ''
            self._bsps = []
            self.bsp_box.configure(values=())
            self.bsp_var.set('')
            self._set_optional_label(self.bsp_message, '')
            self._set_optional_label(self.scenario_error, str(exc))
        self._inputs_changed()

    def _clear_scenario(self):
        self._scenario_file = None
        self._scenario = ''
        self._bsps = []
        self.scenario_var.set('')
        self._set_optional_label(self.scenario_error, '')
        self.bsp_box.configure(values=())
        self.bsp_var.set('')
        self._set_optional_label(self.bsp_message, '')
        self._inputs_changed()

    def estimate_inputs(self) -> EstimateInputs | None:
        if (self._tags is None or form.tags_folder_problem(self._tags) is not None
                or not self._scenario or not self.bsp_var.get()):
            return None
        try:
            values = presets.parse_advanced({
                name: self.advanced_vars[name].get() for name in ('texels_per_unit', 'page_max')
            })
        except ValueError:
            return None
        if len(values) != 2:
            return None
        return EstimateInputs(self._tags, self._scenario, self.bsp_var.get(),
                              values['texels_per_unit'], values['page_max'])

    def _inputs_changed(self):
        # (a preset writes its fields one at a time; _select_preset estimates once when all are in)
        if self.busy or self._setting_preset:
            return
        self._reset_used_rows()
        inputs = self.estimate_inputs()
        if inputs is None:
            if self._estimate_inputs is not None:
                self.generation += 1
            self._estimate_inputs = None
            self._estimate_pending = None
            self.estimate_state = 'idle'
            self.memory_label.configure(text='')
        elif inputs != self._estimate_inputs:
            self.generation += 1
            self._estimate_inputs = inputs
            self._estimate_pending = (self.generation, inputs)
            self.estimate_state = 'calculating'
            self.memory_label.configure(text='Calculating…')
            self._start_estimate()
        self._update_bake_state()

    def _start_estimate(self):
        if self._estimate_running is not None or self._estimate_pending is None:
            return
        generation, inputs = self._estimate_pending
        self._estimate_pending = None
        self._estimate_running = (generation, inputs)
        estimate, events = self._estimate, self.events

        def calculate():
            try:
                result = estimate(inputs)
            except Exception as exc:
                result = exc
            events.put(('estimate', generation, inputs, result))

        # only plain inputs and the queue cross threads; Tk-owning objects stay on the UI thread.
        threading.Thread(target=calculate, daemon=True).start()

    @staticmethod
    def _set_optional_label(label, text):
        label.configure(text=text)
        if text:
            label.grid()
        else:
            label.grid_remove()

    def request(self) -> runner.BakeRequest:
        if self._tags is None:
            raise ValueError('Pick a tags folder.')
        problem = form.tags_folder_problem(self._tags)
        if problem:
            raise ValueError(problem)
        if not self._scenario or not self._bsps:
            raise ValueError(self.scenario_error['text'] or self.bsp_message['text'] or 'Pick a scenario with a BSP.')
        bsp = self.bsp_var.get()
        if not bsp:
            raise ValueError('Pick a BSP.')
        if self.blender is None:
            raise ValueError(self.blender_label['text'])
        if self.preset_var.get() == presets.CUSTOM:
            for field in presets.ADVANCED:
                if not self.advanced_vars[field].get().strip():
                    message = f'{presets.LABELS[field]} is empty. Fill it in, or pick a preset.'
                    self.advanced_error.configure(text=message)
                    raise ValueError(message)
        try:
            advanced = presets.parse_advanced({
                name: var.get()
                for name, var in self.advanced_vars.items()
            })
        except ValueError as exc:
            self.advanced_error.configure(text=str(exc))
            raise
        self.advanced_error.configure(text='')
        settings = presets.build_settings(self.preset_var.get(), advanced, str(self.blender.path),
                                         self._device_value())
        settings.lightmap_32bit = self.lightmap_32bit_var.get()
        return runner.BakeRequest(self._tags, self._scenario, bsp, settings, self.new_names_var.get(),
                                  verbose=self.verbose_var.get(), backups=self.backups_var.get(),
                                  charts=runner.chart_cache_path(self._tags, self._scenario, bsp))

    def remember(self):
        advanced = {
            name: value.get() for name, value in self.advanced_vars.items()
            if value.get().strip()
        }
        self.settings = usersettings.UserSettings(
            tags=self.tags_entry.get(), scenario=self._scenario, bsp=self.bsp_var.get(),
            blender=str(self.blender.path) if self.blender else self.settings.blender,
            preset=self.preset_var.get(), new_names=self.new_names_var.get(), advanced=advanced,
            backups=self.backups_var.get(), lightmap_32bit=self.lightmap_32bit_var.get(),
            advanced_open=self._advanced_open,
            verbose=self.verbose_var.get(), device=self._device_value(),
        )
        usersettings.save(self.settings, self.store_path)

    def _update_bake_state(self):
        valid_tags = self._tags is not None and form.tags_folder_problem(self._tags) is None
        valid = (valid_tags and bool(self._scenario) and bool(self.bsp_var.get()) and self.blender is not None
                 and not self.busy and self.estimate_state in ('idle', 'fits', 'warning', 'error'))
        self.bake_button.configure(state='normal' if valid else 'disabled')

    def _update_new_names_hint(self):
        text = ('Saves the lit level as <name>_lit and leaves your tags as they are.'
                if self.new_names_var.get()
                else ("Replaces your level's tags. Backups go in a backups folder next to them."
                      if self.backups_var.get()
                      else "Replaces your level's tags. No backups are kept."))
        self.new_names_hint.configure(text=text)

    def _update_hint(self):
        self.preset_hint.configure(text=presets.PRESET_HINTS.get(self.preset_var.get(), ''))
        self._set_optional_label(self.preset_note, presets.GPU_NOTE if self.preset_var.get() in presets.PRESET_NAMES else '')

    @staticmethod
    def _device_label(value):
        return next((label for label, setting in presets.DEVICES if setting == value), presets.DEVICES[0][0])

    def _device_value(self):
        return dict(presets.DEVICES).get(self.device_var.get(), 'auto')

    def _show_stage_rows(self, rows):
        for (name, state, detail), (stage, where, text) in zip(self.stage_rows, rows):
            name.configure(text=stage)
            state.configure(text=where or '…', foreground={'GPU': 'green', 'CPU': 'red'}.get(where, ''))
            detail.configure(text=text)

    def _reset_used_rows(self):
        if self._using_stage_rows and not self.busy:
            self._update_device_hint()

    def _update_device_hint(self, *, prediction=True):
        if prediction:
            self._using_stage_rows = False
        if not self._using_stage_rows:
            self._show_stage_rows(devices.stage_rows(self._device_value(), self._adapters, self._gpu_reason, self._denoiser))
        self.gpu_label.configure(text=devices.gpu_text(self._adapters, self._gpu_reason))
        self.device_hint.configure(text=devices.device_hint(self._device_value(), self._adapters, self._gpu_reason))

    def _select_preset(self, name):
        self._setting_preset = True
        try:
            self.preset_var.set(name)
            self._update_hint()
            if name in presets.PRESET_NAMES:
                for field, value in presets.PRESETS[name].items():
                    self.advanced_vars[field].set(presets.format_density(value) if field == 'texels_per_unit' else str(value))
        finally:
            self._setting_preset = False
        self._inputs_changed()

    def _toggle_advanced(self):
        if self._advanced_open:
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text='Advanced ▸')
        else:
            self.advanced_frame.grid()
            self.advanced_button.configure(text='Advanced ▾')
        self._advanced_open = not self._advanced_open

    def _advanced_value_changed(self, name):
        if name == 'texels_per_unit':
            self.advanced_entries[name].configure(values=presets.density_values(self.advanced_vars[name].get().strip()))
        if name == 'page_max':
            self.page_max_hint.configure(text=presets.page_max_hint(self.advanced_vars['page_max'].get()))
        if self._setting_preset:
            return
        selected = self.preset_var.get()
        if selected in presets.PRESET_NAMES:
            matches = True
            for field in presets.ADVANCED:
                text = self.advanced_vars[field].get().strip()
                if not text:
                    matches = False
                    break
                try:
                    value = presets.parse_advanced({field: text})[field]
                    expected = presets.PRESETS[selected][field]
                except ValueError:
                    value = text
                    expected = str(presets.PRESETS[selected][field])
                if value != expected:
                    matches = False
                    break
            if not matches:
                self.preset_var.set(presets.CUSTOM)
                self._update_hint()

    def _browse_tags(self):
        path = filedialog.askdirectory()
        if path:
            self.set_tags(path)

    def _browse_scenario(self):
        if self._scenario_file is not None:
            initial = self._scenario_file.parent
        elif self._tags is not None and (self._tags / 'levels').is_dir():
            initial = self._tags / 'levels'
        else:
            initial = self._tags if self._tags is not None else ''
        path = filedialog.askopenfilename(initialdir=str(initial), filetypes=[('Scenario', '*.scenario')])
        if path:
            try:
                self.set_scenario_file(Path(path))
            except Exception as exc:
                self._set_optional_label(self.scenario_error, str(exc))

    def _discover_blender(self):
        remembered = self.settings.blender or None
        find_blender = self.find_blender
        events = self.events

        def discover():
            try:
                result = find_blender(remembered)
            except blender_find.BlenderNotFound as exc:
                result = exc
            # Tk calls stay on the UI thread; workers only hand results to the poller.
            events.put(('blender', 'discovery', result))

        threading.Thread(target=discover, daemon=True).start()

    def _discover_gpu(self):
        probe = self._probe_gpu
        events = self.events

        def discover():
            try:
                adapters, reason = probe()
            except Exception as error:
                adapters, reason = [], str(error)
            events.put(('gpu', adapters, reason))

        threading.Thread(target=discover, daemon=True).start()

    def _discover_denoiser(self):
        probe, events = self._probe_denoiser, self.events

        def discover():
            try:
                result = probe()
            except Exception:
                result = {}
            events.put(('denoiser', result))

        threading.Thread(target=discover, daemon=True).start()

    def _choose_blender(self):
        path = filedialog.askopenfilename(filetypes=[('Blender', 'blender.exe')])
        if not path:
            return
        self._set_blender_status('Checking Blender…')
        self._update_bake_state()
        events = self.events
        version_finder = blender_find.blender_version
        minimum_version = blender_find.MIN_VERSION

        def check():
            version = version_finder(Path(path))
            if version is None:
                result = blender_find.BlenderNotFound(
                    "That file doesn't look like Blender. Pick blender.exe from the Blender folder."
                )
            elif version < minimum_version:
                result = blender_find.BlenderNotFound(
                    f'That Blender is {version[0]}.{version[1]}. OpenTool needs Blender 5.2 or newer.'
                )
            else:
                result = blender_find.Blender(Path(path), version)
            # Keep Tcl-owning objects out of worker closures; their finalizers must run on the UI thread.
            events.put(('blender', 'change', result))

        threading.Thread(target=check, daemon=True).start()

    def _set_blender_status(self, text):
        self.blender_label.configure(text=text)

    def _poll(self):
        # reschedule whatever a handler raises, or the form would stay locked with nobody listening.
        try:
            self._drain()
        finally:
            try:
                self.root.after(100, self._poll)
            except tk.TclError:
                pass

    def _drain(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'estimate':
                generation, inputs, result = event[1:]
                if self._estimate_running == (generation, inputs):
                    self._estimate_running = None
                if generation != self.generation:
                    self._start_estimate()
                    continue
                try:
                    text, state = estimate_module.describe(result, inputs.page_max, free_physical_bytes())
                except Exception as exc:
                    result = exc
                    text, state = errors.explain(exc), 'error'
                self.memory_label.configure(text=text)
                self.estimate_state = state
                if isinstance(result, BaseException):
                    self._write_exception(result)
                self._update_bake_state()
            elif event[0] == 'blender':
                source, result = event[1:]
                if source == 'discovery' and self._user_chose_blender:
                    continue
                if isinstance(result, blender_find.BlenderNotFound):
                    if source == 'discovery':
                        self.blender = None
                    self._set_blender_status(str(result))
                else:
                    if source == 'change':
                        self._user_chose_blender = True
                    self.blender = result
                    self.settings.blender = str(result.path)
                    self._set_blender_status(f'Blender {result.version[0]}.{result.version[1]} ({result.path})')
                self._update_bake_state()
            elif event[0] == 'gpu':
                self._adapters, self._gpu_reason = event[1:]
                try:
                    self._update_device_hint(prediction=False)
                except tk.TclError:
                    pass
            elif event[0] == 'denoiser':
                self._denoiser = event[1]
                try:
                    self._update_device_hint(prediction=False)
                except tk.TclError:
                    pass
            elif event[0] == 'device':
                device = str(event[1])
                self._set_optional_label(
                    self.device_label,
                    'Baking on the CPU. This can take a long time.' if device.upper() == 'CPU' else f'Baking on {device}')
            elif event[0] == 'log':
                self._append_log(str(event[1]))
                self._live_warnings.add(str(event[1]))
            elif event[0] == 'stage':
                self.stage = event[1]
                if self.close_after_write and event[1] == 'validate':
                    self._finish_close_after_write()
                if not self.close_after_write:
                    self.stage_label.configure(text=STAGE_LABELS[event[1]])
                self.progress['value'] = fraction(event[1]) * 1000
            elif event[0] == 'pages':
                _, done, total = event
                if not self.close_after_write:
                    self.stage_label.configure(
                        text=f'Baking in Blender: page {done} of {total}' if total else f'Baking in Blender: page {done}')
                self.progress['value'] = fraction('blender', done, total) * 1000
            elif event[0] == 'done':
                self._bake_done(event[1])
                if self.close_after_write:
                    self._finish_close_after_write()

    def _append_log(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        extra = int(self.log_box.index('end-1c').split('.')[0]) - 1 - 5000
        if extra > 0:
            self.log_box.delete('1.0', f'{extra + 1}.0')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _bake_done(self, outcome):
        self.busy = False
        self.stage = None
        self.progress['value'] = 1000
        self.stage_label.configure(text=outcome.message,
                                   foreground='red' if outcome.status == 'failed' else '')
        if outcome.problems:
            self._append_log('Problems')
            for problem in outcome.problems:
                self._append_log(str(problem))
        if outcome.devices:
            self._using_stage_rows = True
            self._show_stage_rows(devices.stage_rows_used(outcome.devices))
            self._set_optional_label(self.device_label, devices.result_line(outcome.devices, outcome.rays, outcome.warnings))
        else:
            self._update_device_hint()
        for warning in outcome.warnings:
            # (the runner already sent most warnings as they happened)
            if f'WARNING: {warning}' not in self._live_warnings:
                self._append_log(f'WARNING: {warning}')
        self._output_folder = outcome.output_folder
        self.open_button.configure(state='normal' if outcome.output_folder else 'disabled')
        self._unlock_form()

    def _lock_form(self):
        self._locked = {}
        keep = (self.bake_button, self.open_button, self.advanced_button)
        pending = [self.frame]
        while pending:
            for widget in pending.pop().winfo_children():
                pending.append(widget)
                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)) and widget not in keep:
                    self._locked[widget] = str(widget.cget('state'))
                    widget.configure(state='disabled')

    def _unlock_form(self):
        for widget, state in self._locked.items():
            widget.configure(state=state)
        self._locked = {}
        self._update_bake_state()

    def start_bake(self):
        if self.busy:
            return
        try:
            request = self.request()
        except ValueError as exc:
            self.stage_label.configure(text=str(exc), foreground='red')
            return
        if self.estimate_state == 'warning':
            message = self.memory_label['text'].removesuffix(estimate_module.RISK_SUFFIX)
            if not messagebox.askyesno(
                    'Not enough memory',
                    message + '\n\nBake anyway? Blender may run out of memory and stop.'):
                return
        if not request.new_names:
            tail = ('Backups go in a backups folder next to them.' if request.backups
                    else 'No backups are kept.')
            if not messagebox.askyesno(
                    'Replace your tags?',
                    f'This replaces {request.bsp} and its lightmaps in place. {tail} Continue?'):
                return
        try:
            self.remember()
        except Exception as exc:
            self._write_exception(exc)
        self.busy = True
        self.stage = None
        self._live_warnings = set()
        self.close_after_write = False
        self._output_folder = None
        self.bake_button.configure(state='disabled')
        self.open_button.configure(state='disabled')
        self._lock_form()
        self.log_box.configure(state='normal')
        self.log_box.delete('1.0', 'end')
        self.log_box.configure(state='disabled')
        self._set_optional_label(self.device_label, '')
        self._update_device_hint()
        self.stage_label.configure(text='Starting…', foreground='')
        self.progress['value'] = 0
        gate = self._gate = _BakeGate()
        put = self.events.put

        def emit(event):
            if event[0] == 'stage' and event[1] == 'write':
                with gate.lock:
                    closing = gate.closing
                    if not closing:
                        gate.writing = True
                        put(event)
                if closing:
                    # the window is closing and the process is about to exit; never start a write.
                    threading.Event().wait()
                return
            if event[0] == 'done' or event == ('stage', 'validate'):
                with gate.lock:
                    gate.writing = False
            put(event)

        # the thread gets the runner and the gate, not self, so no Tk object is finalized off the UI thread.
        threading.Thread(target=self._run_bake, args=(request, emit), daemon=True).start()

    def _write_exception(self, exc, tb=None):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as log:
            traceback.print_exception(type(exc), exc, tb or exc.__traceback__, file=log)

    def _report_callback_exception(self, exc, value, tb):
        self._write_exception(value, tb)
        self.stage_label.configure(text=f'Something went wrong in OpenTool. The details are in {self.log_path}.')

    def _open_output(self):
        if self._output_folder is not None:
            os.startfile(self._output_folder)

    def on_close(self):
        if not self.busy:
            self._close_window()
            return
        if not messagebox.askyesno(
                'Stop the bake?',
                "A bake is running. Closing OpenTool stops it, and no tags will be written "
                "if it hasn't reached that step yet."):
            return
        with self._gate.lock:
            self._gate.closing = True
            writing = self._gate.writing
        if not self.busy:
            self._close_window()
        elif writing:
            # the pipeline writes the bitmap, BSP and scenario one after another; stopping between
            # them would leave a lit BSP whose scenario was never written.
            self.close_after_write = True
            self.stage_label.configure(text='Finishing writing tags…')
        else:
            self._exit_now()

    def _finish_close_after_write(self):
        self.close_after_write = False
        self._exit_now()

    def _close_window(self):
        try:
            try:
                self.remember()
            except Exception as exc:
                self._write_exception(exc)
        finally:
            self.window.destroy()

    def _exit_now(self):
        self._close_window()
        # os._exit skips mainloop teardown; the job object then ends Blender with the process.
        os._exit(0)


def main() -> int:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    if sys.stdout is None:
        sys.stdout = open(os.devnull, 'w')
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')
    jobobject.kill_children_on_exit()
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0
