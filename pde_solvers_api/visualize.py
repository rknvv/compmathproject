import time
import requests
import threading
import ipywidgets as widgets

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from plotly.subplots import make_subplots


DEFAULT_SERVER_URL = "http://localhost:8000"
PRESETS = {
    "Пятна": {"F": 0.025, "k": 0.06, "Du": 0.16, "Dv": 0.08},
    "Волны": {"F": 0.014, "k": 0.054, "Du": 0.16, "Dv": 0.08},
    "Спирали": {"F": 0.018, "k": 0.051, "Du": 0.16, "Dv": 0.08},
    "Лабиринт": {"F": 0.029, "k": 0.057, "Du": 0.16, "Dv": 0.08},
    "Дырки": {"F": 0.039, "k": 0.058, "Du": 0.16, "Dv": 0.08},
    "Хаос": {"F": 0.026, "k": 0.051, "Du": 0.16, "Dv": 0.08},
    "Мир U-коньков": {"F": 0.062, "k": 0.061, "Du": 0.16, "Dv": 0.08},
    "Тьюринговские узоры": {"F": 0.030, "k": 0.055, "Du": 0.16, "Dv": 0.08},
    "Клетки": {"F": 0.018, "k": 0.055, "Du": 0.16, "Dv": 0.08},
    "Цветы": {"F": 0.055, "k": 0.062, "Du": 0.16, "Dv": 0.08},
    "Солитоны": {"F": 0.026, "k": 0.061, "Du": 0.16, "Dv": 0.08},
}


class GrayScottVisualizer:
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url
        self.U = None
        self.V = None
        self.running = False
        self.sim_thread = None
        self.vis_mode = "2D"
        self.fig_widget = None

        self.params = {
            "Du": 0.16,
            "Dv": 0.08,
            "F": 0.055,
            "k": 0.062,
            "grid_size": 100,
            "dt": 1.0,
        }

        self.steps_per_update = 10

        self._create_widgets()
        self._create_layout()

        self.output_widget = widgets.Output()

    def _create_widgets(self):
        self.title = widgets.HTML(
            "<h2 style='text-align: center; color: #3366cc;'>Визуализация системы Грея-Скотта</h2>"
        )

        self.du_slider = widgets.FloatSlider(
            value=self.params["Du"],
            min=0.01,
            max=0.5,
            step=0.01,
            description="Du:",
            continuous_update=False,
            style={"description_width": "40px"},
        )
        self.dv_slider = widgets.FloatSlider(
            value=self.params["Dv"],
            min=0.01,
            max=0.5,
            step=0.01,
            description="Dv:",
            continuous_update=False,
            style={"description_width": "40px"},
        )

        self.f_slider = widgets.FloatSlider(
            value=self.params["F"],
            min=0.01,
            max=0.1,
            step=0.001,
            description="F:",
            continuous_update=False,
            style={"description_width": "40px"},
        )
        self.k_slider = widgets.FloatSlider(
            value=self.params["k"],
            min=0.01,
            max=0.1,
            step=0.001,
            description="k:",
            continuous_update=False,
            style={"description_width": "40px"},
        )

        self.dt_slider = widgets.FloatSlider(
            value=self.params["dt"],
            min=0.1,
            max=2.0,
            step=0.1,
            description="dt:",
            continuous_update=False,
            style={"description_width": "40px"},
        )
        self.grid_size_slider = widgets.IntSlider(
            value=self.params["grid_size"],
            min=50,
            max=200,
            step=10,
            description="Сетка:",
            continuous_update=False,
            style={"description_width": "40px"},
        )
        self.steps_slider = widgets.IntSlider(
            value=self.steps_per_update,
            min=1,
            max=50,
            step=1,
            description="Шагов:",
            continuous_update=False,
            style={"description_width": "50px"},
        )

        button_style = {"button_width": "120px", "description_width": "0px"}

        self.init_button = widgets.Button(
            description="Инициализация",
            button_style="primary",
            layout=widgets.Layout(width="150px"),
            style=button_style,
        )
        self.step_button = widgets.Button(
            description="Шаг",
            button_style="info",
            layout=widgets.Layout(width="100px"),
            style=button_style,
        )
        self.start_button = widgets.Button(
            description="Старт",
            button_style="success",
            layout=widgets.Layout(width="100px"),
            style=button_style,
        )
        self.stop_button = widgets.Button(
            description="Стоп",
            button_style="danger",
            layout=widgets.Layout(width="100px"),
            style=button_style,
        )

        self.status_indicator = widgets.HTML(
            value="<span style='color: gray;'>Ожидание инициализации...</span>"
        )

        self.vis_mode_dropdown = widgets.Dropdown(
            options=["2D", "3D"],
            value="2D",
            description="Режим:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        self.speed_dropdown = widgets.Dropdown(
            options=[
                ("Медленно", 0.2),
                ("Средне", 0.1),
                ("Быстро", 0.05),
                ("Очень быстро", 0.01),
            ],
            value=0.05,
            description="Скорость:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )

        self.preset_dropdown = widgets.Dropdown(
            options=list(PRESETS.keys()),
            value=list(PRESETS.keys())[0],
            description="Шаблон:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )

        self.method_dropdown = widgets.Dropdown(
            options=["crank_nicolson", "runge_kutta"],
            value="crank_nicolson",
            description="Метод:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="150px"),
        )

        self.init_button.on_click(self._initialize_handler)
        self.step_button.on_click(self._step_handler)
        self.start_button.on_click(self._start_handler)
        self.stop_button.on_click(self._stop_handler)
        self.vis_mode_dropdown.observe(self._vis_mode_changed, names="value")
        self.preset_dropdown.observe(self._preset_changed, names="value")

        for slider, param in [
            (self.du_slider, "Du"),
            (self.dv_slider, "Dv"),
            (self.f_slider, "F"),
            (self.k_slider, "k"),
            (self.dt_slider, "dt"),
            (self.grid_size_slider, "grid_size"),
        ]:

            def make_handler(p=param):
                def handler(change):
                    self.params[p] = change.new

                return handler

            slider.observe(make_handler(), names="value")

        self.steps_slider.observe(
            lambda change: setattr(self, "steps_per_update", change.new), names="value"
        )

    def _create_layout(self):
        diff_title = widgets.HTML("<h4>Параметры диффузии</h4>")
        react_title = widgets.HTML("<h4>Параметры реакции</h4>")
        sim_title = widgets.HTML("<h4>Параметры симуляции</h4>")
        controls_title = widgets.HTML("<h4>Управление</h4>")

        diff_box = widgets.VBox(
            [
                diff_title,
                widgets.HBox([self.du_slider]),
                widgets.HBox([self.dv_slider]),
            ],
            layout=widgets.Layout(
                border="1px solid #ddd", padding="10px", margin="5px", width="300px"
            ),
        )
        react_box = widgets.VBox(
            [react_title, widgets.HBox([self.f_slider]), widgets.HBox([self.k_slider])],
            layout=widgets.Layout(
                border="1px solid #ddd", padding="10px", margin="5px", width="300px"
            ),
        )
        sim_box = widgets.VBox(
            [
                sim_title,
                widgets.HBox([self.dt_slider]),
                widgets.HBox([self.grid_size_slider]),
                widgets.HBox([self.steps_slider]),
            ],
            layout=widgets.Layout(
                border="1px solid #ddd", padding="10px", margin="5px", width="300px"
            ),
        )
        control_box = widgets.VBox(
            [
                controls_title,
                widgets.HBox(
                    [
                        self.init_button,
                        self.step_button,
                        self.start_button,
                        self.stop_button,
                    ]
                ),
                widgets.HBox([self.vis_mode_dropdown, self.speed_dropdown]),
                widgets.HBox([self.preset_dropdown, self.method_dropdown]),
                self.status_indicator,
            ],
            layout=widgets.Layout(
                border="1px solid #ddd", padding="10px", margin="5px"
            ),
        )

        left_panel = widgets.VBox(
            [diff_box, react_box, sim_box], layout=widgets.Layout(width="320px")
        )
        right_panel = widgets.VBox([control_box])
        self.main_container = widgets.VBox(
            [self.title, widgets.HBox([left_panel, right_panel])]
        )

    def initialize(self) -> None:
        try:
            self.status_indicator.value = (
                "<span style='color: blue;'>Инициализация...</span>"
            )
            response = requests.post(
                f"{self.server_url}/gray-scott/initialize", json=self.params
            )
            response.raise_for_status()
            data = response.json()
            self.U = np.array(data["U"])
            self.V = np.array(data["V"])

            if self.vis_mode == "2D":
                fig = make_subplots(
                    rows=1, cols=2, subplot_titles=("Концентрация U", "Концентрация V")
                )
                fig.add_trace(
                    go.Contour(
                        z=self.U, colorscale="YlGnBu", zmin=0, zmax=1, showscale=True
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Contour(
                        z=self.V, colorscale="YlOrRd", zmin=0, zmax=1, showscale=True
                    ),
                    row=1,
                    col=2,
                )
                fig.update_layout(
                    height=500,
                    width=950,
                    title_text=f"F={self.params['F']:.4f}, k={self.params['k']:.4f}",
                    margin=dict(l=50, r=50, t=80, b=50),
                )
            else:
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "surface"}, {"type": "surface"}]],
                    subplot_titles=("Концентрация U", "Концентрация V"),
                )
                x = np.arange(self.U.shape[0])
                y = np.arange(self.U.shape[1])
                X, Y = np.meshgrid(x, y)
                fig.add_trace(
                    go.Surface(z=self.U, x=X, y=Y, colorscale="YlGnBu", cmin=0, cmax=1),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Surface(z=self.V, x=X, y=Y, colorscale="YlOrRd", cmin=0, cmax=1),
                    row=1,
                    col=2,
                )
                fig.update_layout(
                    height=500,
                    width=950,
                    title_text=f"F={self.params['F']:.4f}, k={self.params['k']:.4f}",
                    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="U"),
                    scene2=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="V"),
                    margin=dict(l=50, r=50, t=80, b=50),
                )

            self.fig_widget = go.FigureWidget(fig)
            with self.output_widget:
                self.output_widget.clear_output(wait=True)
                display(self.fig_widget)

            self.status_indicator.value = (
                "<span style='color: green;'>Инициализировано</span>"
            )
        except Exception as e:
            error_msg = f"Ошибка инициализации: {e}"
            self.status_indicator.value = (
                f"<span style='color: red;'>{error_msg}</span>"
            )
            print(error_msg)

    def step(self, steps: int = None) -> None:
        if self.U is None or self.V is None:
            self.status_indicator.value = (
                "<span style='color: orange;'>Сначала инициализируйте симуляцию</span>"
            )
            return

        if steps is None:
            steps = self.steps_per_update

        try:
            self.status_indicator.value = (
                "<span style='color: blue;'>Выполняем шаг...</span>"
            )
            input_data = {
                "params": self.params,
                "U": self.U.tolist(),
                "V": self.V.tolist(),
                "steps": steps,
            }
            method = self.method_dropdown.value
            response = requests.post(
                f"{self.server_url}/gray-scott/step/{method}", json=input_data
            )
            response.raise_for_status()
            data = response.json()
            self.U = np.array(data["U"])
            self.V = np.array(data["V"])

            with self.fig_widget.batch_update():
                if self.vis_mode == "2D":
                    self.fig_widget.data[0].z = self.U
                    self.fig_widget.data[1].z = self.V
                else:
                    self.fig_widget.data[0].z = self.U
                    self.fig_widget.data[1].z = self.V
                self.fig_widget.layout.title.text = (
                    f"F={self.params['F']:.4f}, k={self.params['k']:.4f}"
                )

            self.status_indicator.value = (
                "<span style='color: green;'>Шаг выполнен</span>"
            )
        except Exception as e:
            error_msg = f"Ошибка при выполнении шага: {e}"
            self.status_indicator.value = (
                f"<span style='color: red;'>{error_msg}</span>"
            )
            print(error_msg)
            self.running = False

    def run_simulation(self) -> None:
        self.running = True
        self.status_indicator.value = (
            "<span style='color: green;'>Симуляция запущена</span>"
        )

        while self.running:
            try:
                if self.U is None or self.V is None:
                    self.status_indicator.value = "<span style='color: orange;'>Сначала инициализируйте симуляцию</span>"
                    self.running = False
                    break

                steps = self.steps_per_update
                input_data = {
                    "params": self.params,
                    "U": self.U.tolist(),
                    "V": self.V.tolist(),
                    "steps": steps,
                }
                method = self.method_dropdown.value
                response = requests.post(
                    f"{self.server_url}/gray-scott/step/{method}", json=input_data
                )
                response.raise_for_status()
                data = response.json()
                self.U = np.array(data["U"])
                self.V = np.array(data["V"])

                with self.fig_widget.batch_update():
                    if self.vis_mode == "2D":
                        self.fig_widget.data[0].z = self.U
                        self.fig_widget.data[1].z = self.V
                    else:
                        self.fig_widget.data[0].z = self.U
                        self.fig_widget.data[1].z = self.V
                    self.fig_widget.layout.title.text = (
                        f"F={self.params['F']:.4f}, k={self.params['k']:.4f}"
                    )

                time.sleep(self.speed_dropdown.value)
            except Exception as e:
                self.status_indicator.value = (
                    f"<span style='color: red;'>Ошибка: {e}</span>"
                )
                print(f"Ошибка в симуляции: {e}")
                self.running = False
                break

        self.status_indicator.value = (
            "<span style='color: blue;'>Симуляция остановлена</span>"
        )

    def display(self) -> None:
        display(widgets.VBox([widgets.HBox([self.main_container]), self.output_widget]))

    def _initialize_handler(self, _) -> None:
        self.initialize()

    def _step_handler(self, _) -> None:
        self.step()

    def _start_handler(self, _) -> None:
        if not self.running:
            self.sim_thread = threading.Thread(target=self.run_simulation)
            self.sim_thread.daemon = True
            self.sim_thread.start()

    def _stop_handler(self, _) -> None:
        self.running = False
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)

    def _vis_mode_changed(self, change) -> None:
        self.vis_mode = change.new
        if self.U is not None and self.V is not None:
            with self.output_widget:
                self.output_widget.clear_output(wait=True)
                if self.vis_mode == "2D":
                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Концентрация U", "Концентрация V"),
                    )
                    fig.add_trace(
                        go.Contour(
                            z=self.U,
                            colorscale="YlGnBu",
                            zmin=0,
                            zmax=1,
                            showscale=True,
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Contour(
                            z=self.V,
                            colorscale="YlOrRd",
                            zmin=0,
                            zmax=1,
                            showscale=True,
                        ),
                        row=1,
                        col=2,
                    )
                    fig.update_layout(
                        height=500,
                        width=950,
                        title_text=f"F={self.params['F']:.4f}, k={self.params['k']:.4f}",
                        margin=dict(l=50, r=50, t=80, b=50),
                    )
                else:
                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        specs=[[{"type": "surface"}, {"type": "surface"}]],
                        subplot_titles=("Концентрация U", "Концентрация V"),
                    )
                    x = np.arange(self.U.shape[0])
                    y = np.arange(self.U.shape[1])
                    X, Y = np.meshgrid(x, y)
                    fig.add_trace(
                        go.Surface(
                            z=self.U, x=X, y=Y, colorscale="YlGnBu", cmin=0, cmax=1
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Surface(
                            z=self.V, x=X, y=Y, colorscale="YlOrRd", cmin=0, cmax=1
                        ),
                        row=1,
                        col=2,
                    )
                    fig.update_layout(
                        height=500,
                        width=950,
                        title_text=f"F={self.params['F']:.4f}, k={self.params['k']:.4f}",
                        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="U"),
                        scene2=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="V"),
                        margin=dict(l=50, r=50, t=80, b=50),
                    )
                self.fig_widget = go.FigureWidget(fig)
                display(self.fig_widget)

    def _preset_changed(self, change) -> None:
        preset = PRESETS[change.new]
        self.params.update(preset)
        self.du_slider.value = preset["Du"]
        self.dv_slider.value = preset["Dv"]
        self.f_slider.value = preset["F"]
        self.k_slider.value = preset["k"]
        self.initialize()
