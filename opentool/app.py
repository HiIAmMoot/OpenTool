import ctypes
import os
from pathlib import Path
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

from lightmapper import blender_find
from . import form, jobobject, presets, runner, usersettings


class App:
    def __init__(self, root: tk.Misc, *, store_path: Path | None = None,
                 find_blender=blender_find.find_blender, run_bake=runner.run_bake,
                 log_path: Path | None = None):
        self.root = root
        self.window = root.winfo_toplevel()
        self.tk_root = root._root()
        self.store_path = store_path
        self.log_path = Path(log_path) if log_path is not None else Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'OpenTool' / 'opentool.log'
        self.find_blender = find_blender
        self.run_bake = run_bake
        self.events: queue.Queue = queue.Queue()
        self.blender: blender_find.Blender | None = None
        self._user_chose_blender = False
        self._tags: Path | None = None
        self._scenario = ''
        self._scenario_file: Path | None = None
        self._bsps: list[str] = []
        self.settings = usersettings.load(store_path)

        self.window.title('OpenTool lightmapper')
        self.window.minsize(640, 560)
        self.window.resizable(True, True)
        self.tk_root.report_callback_exception = self._report_callback_exception
        self.window.protocol('WM_DELETE_WINDOW', self._close)
        self._build()
        self._restore()
        self._set_blender_status('Looking for Blender…')
        self._discover_blender()
        root.after(100, self._poll)

    def _build(self):
        self.frame = ttk.Frame(self.window, padding=12)
        self.frame.pack(fill='both', expand=True)
        self.frame.columnconfigure(0, weight=1)

        tags_row = ttk.Frame(self.frame)
        tags_row.grid(row=0, column=0, sticky='ew')
        tags_row.columnconfigure(0, weight=1)
        ttk.Label(tags_row, text='Tags folder').grid(row=0, column=0, sticky='w')
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
        self.bsp_box.bind('<<ComboboxSelected>>', lambda _event: self._update_bake_state())
        self.bsp_message = ttk.Label(self.frame)
        self.bsp_message.grid(row=5, column=0, sticky='w')
        self.bsp_message.grid_remove()

        quality = ttk.Frame(self.frame)
        quality.grid(row=6, column=0, sticky='ew', pady=(12, 0))
        ttk.Label(quality, text='Quality').pack(side='left')
        self.preset_var = tk.StringVar(value='Normal')
        for name in presets.PRESET_NAMES:
            ttk.Radiobutton(quality, text=name, value=name, variable=self.preset_var,
                            command=self._update_hint).pack(side='left', padx=(8, 0))
        self.preset_hint = ttk.Label(quality, text=presets.PRESET_HINTS['Normal'])
        self.preset_hint.pack(side='left', padx=(12, 0))

        self.new_names_var = tk.BooleanVar(value=self.settings.new_names)
        ttk.Checkbutton(self.frame, text='Write to new tag names', variable=self.new_names_var,
                        command=self._update_bake_state).grid(row=7, column=0, sticky='w', pady=(8, 0))
        ttk.Label(self.frame, text='Saves the lit level as <name>_lit and leaves your tags as they are.').grid(row=8, column=0, sticky='w')

        self.advanced_button = ttk.Button(self.frame, text='Advanced ▸', command=self._toggle_advanced)
        self.advanced_button.grid(row=9, column=0, sticky='w', pady=(8, 0))
        self.advanced_frame = ttk.Frame(self.frame)
        self.advanced_frame.grid(row=10, column=0, sticky='ew')
        self.advanced_frame.grid_remove()
        self.advanced_frame.columnconfigure(1, weight=1)
        self.advanced_vars = {}
        self.advanced_entries = {}
        self._advanced_placeholders = {}
        self._advanced_placeholder_values = {}
        self._advanced_open = False
        labels = {'texels_per_unit': 'Texel density', 'bounces': 'Bounces', 'page_max': 'Page size'}
        for index, name in enumerate(presets.ADVANCED):
            ttk.Label(self.advanced_frame, text=labels[name]).grid(row=index, column=0, sticky='w', padx=(0, 8), pady=2)
            value = tk.StringVar()
            self.advanced_vars[name] = value
            if name == 'bounces':
                entry = ttk.Spinbox(self.advanced_frame, from_=0, to=8, textvariable=value)
            elif name == 'page_max':
                entry = ttk.Combobox(self.advanced_frame, values=presets.PAGE_MAX_CHOICES, textvariable=value, state='readonly')
            else:
                entry = ttk.Entry(self.advanced_frame, textvariable=value)
            entry.grid(row=index, column=1, sticky='ew', pady=2)
            self.advanced_entries[name] = entry
            value.trace_add('write', lambda *_args, field=name: self._advanced_value_changed(field))
            entry.bind('<FocusIn>', lambda _event, field=name: self._clear_placeholder(field))
            entry.bind('<FocusOut>', lambda _event, field=name: self._show_placeholder(field))
            self._show_placeholder(name)
        self.advanced_error = ttk.Label(self.advanced_frame, foreground='red')
        self.advanced_error.grid(row=3, column=0, columnspan=2, sticky='w')

        blender_row = ttk.Frame(self.frame)
        blender_row.grid(row=11, column=0, sticky='ew', pady=(12, 0))
        self.blender_label = ttk.Label(blender_row, text='Looking for Blender…')
        self.blender_label.pack(side='left', fill='x', expand=True)
        ttk.Button(blender_row, text='Change…', command=self._choose_blender).pack(side='right')

        controls = ttk.Frame(self.frame)
        controls.grid(row=12, column=0, sticky='ew', pady=(12, 0))
        self.bake_button = ttk.Button(controls, text='Bake', command=lambda: None, state='disabled')
        self.bake_button.pack(side='left')
        self._output_folder: Path | None = None

        self.stage_label = ttk.Label(self.frame, text='Ready')
        self.stage_label.grid(row=13, column=0, sticky='w', pady=(8, 0))
        self.progress = ttk.Progressbar(self.frame, maximum=1000, mode='determinate')
        self.progress.grid(row=14, column=0, sticky='ew')
        self.device_label = ttk.Label(self.frame)
        self.device_label.grid(row=15, column=0, sticky='w')
        self.device_label.grid_remove()
        self.log_box = ScrolledText(self.frame, height=7, state='disabled', wrap='word', font='TkFixedFont')
        self.log_box.grid(row=16, column=0, sticky='nsew', pady=(4, 0))
        self.frame.rowconfigure(16, weight=1)
        self.open_button = ttk.Button(self.frame, text='Open output folder', command=self._open_output, state='disabled')
        self.open_button.grid(row=17, column=0, sticky='e', pady=(4, 0))

    def _restore(self):
        self.preset_var.set(self.settings.preset)
        self._update_hint()
        for name, value in self.settings.advanced.items():
            if name in self.advanced_vars:
                self._clear_placeholder(name)
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
                    self._update_bake_state()
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
        self._update_bake_state()

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
        except (OSError, ValueError, KeyError) as exc:
            self._scenario = ''
            self._bsps = []
            self.bsp_box.configure(values=())
            self.bsp_var.set('')
            self._set_optional_label(self.bsp_message, '')
            self._set_optional_label(self.scenario_error, str(exc))
        self._update_bake_state()

    def _clear_scenario(self):
        self._scenario_file = None
        self._scenario = ''
        self._bsps = []
        self.scenario_var.set('')
        self._set_optional_label(self.scenario_error, '')
        self.bsp_box.configure(values=())
        self.bsp_var.set('')
        self._set_optional_label(self.bsp_message, '')

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
        try:
            advanced = presets.parse_advanced({
                name: '' if self._advanced_placeholders.get(name) else var.get()
                for name, var in self.advanced_vars.items()
            })
        except ValueError as exc:
            self.advanced_error.configure(text=str(exc))
            raise
        self.advanced_error.configure(text='')
        settings = presets.build_settings(self.preset_var.get(), advanced, str(self.blender.path))
        return runner.BakeRequest(self._tags, self._scenario, bsp, settings, self.new_names_var.get())

    def remember(self):
        advanced = {
            name: value.get() for name, value in self.advanced_vars.items()
            if value.get().strip() and not self._advanced_placeholders.get(name)
        }
        self.settings = usersettings.UserSettings(
            tags=self.tags_entry.get(), scenario=self._scenario, bsp=self.bsp_var.get(),
            blender=str(self.blender.path) if self.blender else self.settings.blender,
            preset=self.preset_var.get(), new_names=self.new_names_var.get(), advanced=advanced,
            advanced_open=self._advanced_open,
        )
        usersettings.save(self.settings, self.store_path)

    def _update_bake_state(self):
        valid_tags = self._tags is not None and form.tags_folder_problem(self._tags) is None
        valid = valid_tags and bool(self._scenario) and bool(self.bsp_var.get()) and self.blender is not None
        self.bake_button.configure(state='normal' if valid else 'disabled')

    def _update_hint(self):
        self.preset_hint.configure(text=presets.PRESET_HINTS.get(self.preset_var.get(), ''))
        for name in self.advanced_vars:
            if self._advanced_placeholders.get(name):
                self._advanced_placeholders[name] = False
                self.advanced_vars[name].set('')
                self._show_placeholder(name)

    def _toggle_advanced(self):
        if self._advanced_open:
            self.advanced_frame.grid_remove()
            self.advanced_button.configure(text='Advanced ▸')
        else:
            self.advanced_frame.grid()
            self.advanced_button.configure(text='Advanced ▾')
        self._advanced_open = not self._advanced_open

    def _placeholder_text(self, name):
        settings = presets.build_settings(self.preset_var.get(), {}, str(self.blender.path) if self.blender else '')
        value = getattr(settings, name)
        return str(value)

    def _show_placeholder(self, name):
        variable = self.advanced_vars[name]
        if variable.get():
            return
        placeholder = self._placeholder_text(name)
        self._advanced_placeholders[name] = True
        self._advanced_placeholder_values[name] = placeholder
        variable.set(placeholder)
        self.advanced_entries[name].configure(foreground='gray')

    def _clear_placeholder(self, name):
        if not self._advanced_placeholders.get(name):
            return
        self._advanced_placeholders[name] = False
        self.advanced_vars[name].set('')
        self.advanced_entries[name].configure(foreground='black')

    def _advanced_value_changed(self, name):
        if (self._advanced_placeholders.get(name)
                and self.advanced_vars[name].get() != self._advanced_placeholder_values.get(name)):
            self._advanced_placeholders[name] = False
            self.advanced_entries[name].configure(foreground='black')

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
            except ValueError as exc:
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

    def _choose_blender(self):
        path = filedialog.askopenfilename(filetypes=[('Blender', 'blender.exe')])
        if not path:
            return
        self._user_chose_blender = True
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
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'blender':
                source, result = event[1:]
                if source == 'discovery' and self._user_chose_blender:
                    continue
                if isinstance(result, blender_find.BlenderNotFound):
                    if source == 'discovery':
                        self.blender = None
                    self._set_blender_status(str(result))
                else:
                    self.blender = result
                    self.settings.blender = str(result.path)
                    self._set_blender_status(f'Blender {result.version[0]}.{result.version[1]} ({result.path})')
                self._update_bake_state()
            elif event[0] == 'device':
                self._set_optional_label(self.device_label, str(event[1]))
            elif event[0] == 'log':
                self._append_log(str(event[1]))
        try:
            self.root.after(100, self._poll)
        except tk.TclError:
            pass

    def _append_log(self, text):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

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

    def _close(self):
        try:
            try:
                self.remember()
            except Exception as exc:
                self._write_exception(exc)
        finally:
            self.window.destroy()


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
