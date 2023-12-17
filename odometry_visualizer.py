import numpy as np
import config
import pyqtgraph as pg
import traceback
from queue import Empty
from PyQt6 import QtCore


class Visualizer(pg.GraphicsView):
    def __init__(self, data_queue, horizon_queue, exit_event):
        print("INITED VISUALIZER")
        super(Visualizer, self).__init__()
        self.data_queue = data_queue
        self.horizon_queue = horizon_queue
        self.exit_event = exit_event

        self.backwards_horizon = None
        self.initialized = False

        self.layout = pg.GraphicsLayout()
        self.setCentralWidget(self.layout)

        plot_list = [
            ("velocities", -2.5, 250*config.KM_H, "m/s", [
                ((255, 255, 0), "vx"),
                ((255, 255, 0), "vy")
            ]),
            ("accelerations", -2*9.8, 2*9.8, "m/s", [
                ((255, 255, 0), "ax"),
                ((0, 255, 255), "ay")
            ]),
            ("angulars", -200, 200, "deg/s", [
                ((255, 255, 0), "hdg"),
                ((0, 255, 255), "w"),
                ((255, 0, 255), "dw")
            ]),
            ("controls", -1, 1, "amplitude", [
                ((255, 255, 0), "steer"),
                ((0, 255, 255), "throttle"),
            ]),
            ("mpc duration", 0, 100, "ms", [
                ((255, 255, 0), "duration"),
                ((0, 255, 255), "target")
            ])
        ]

        # Construct graph:
        self.n_plot_items = sum([len(t[4]) for t in plot_list])
        self.plot_history_len = 100 + config.mpc_N
        self.x = np.repeat(np.arange(self.plot_history_len), self.n_plot_items).reshape(self.plot_history_len, self.n_plot_items).T
        self.y = np.zeros([self.n_plot_items, self.plot_history_len])
        self.plots = []
        self.plot_items = []
        plot_i = 0
        item_i = 0
        for title, lb, ub, y_label, items in plot_list:
            self.plots.append(self.layout.addPlot(row=plot_i, col=0))
            self.layout.nextRow()
            self.plots[plot_i].setYRange(lb, ub)
            self.plots[plot_i].setTitle(title)
            self.plots[plot_i].showGrid(True, True)
            self.plots[plot_i].setLabel(axis='left', text=y_label)
            legend = self.plots[plot_i].addLegend()
            legend.setBrush('k')
            legend.setOffset(1)

            # Add a vertical line at the specified index
            vertical_line = pg.InfiniteLine(pos=self.plot_history_len - (config.mpc_N+1), angle=90, movable=False, pen='r')
            self.plots[plot_i].addItem(vertical_line)

            for color, name in items:
                self.plot_items.append(
                    self.plots[plot_i].plot(
                        self.x[item_i], self.y[item_i], pen=pg.mkPen(color=color), name=name
                    )
                )
                item_i += 1
            plot_i += 1

        self.n_plots = plot_i
        for i in range(plot_i):
            self.layout.layout.setRowFixedHeight(i, 240)
            self.layout.layout.setColumnFixedWidth(i, 1080)
        height = int(plot_i*(240+self.layout.layout.rowSpacing(0))) + 20
        self.setFixedHeight(height)
        width = 1080 + int(2*self.layout.layout.columnSpacing(0)) + 20
        self.setFixedWidth(width)

        # Start UI tick:
        self.timer = QtCore.QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.update_data)
        self.timer.start()

    def update_data(self):
        if self.exit_event.is_set():
            self.close()

        try:
            data, solve_time = self.data_queue.get(block=False)

            graph_data = {
                'velocities': (0, [data[4], data[5]]),
                'accelerations': (2, [data[7], data[8]]),
                'angulars': (4, [180/np.pi * data[3], 180/np.pi * data[6], 180/np.pi * data[9]]),
                'controls': (7, [data[10], data[11]]),
                'mpc timing': (9, [solve_time, 1e3*config.mpc_sample_time])
            }

            # Update and draw the odometry/info graph
            for first_i, values_list in graph_data.values():
                for j, value in enumerate(values_list):
                    self.y[first_i+j, 0:-(config.mpc_N+2)] = self.y[first_i+j, 1:-(config.mpc_N+1)]
                    self.y[first_i+j, -(config.mpc_N+2)] = value
                    self.plot_items[first_i+j].setData(self.x[first_i+j], self.y[first_i+j])

        except Empty:
            pass
        except Exception as e:
            print("Graph error: ", traceback.print_exception(e))
            self.close()

        try:
            horizon = self.horizon_queue.get(block=False)

            dt = np.linspace(config.mpc_sample_time, config.nonuniform_sample_low, config.mpc_N + 1)
            # dt = np.tile(config.mpc_sample_time, config.mpc_N+1)
            diff = np.r_[horizon[1:, 3:6] - horizon[:-1, 3:6], np.zeros([1, 3])]
            accels = (diff.T / dt).T

            graph_data = {
                'velocities': (0, [horizon[:, 3], horizon[:, 4]]),
                'accelerations': (2, [
                    accels[:, 0] - horizon[:, 4]*horizon[:, 5],
                    accels[:, 1] + horizon[:, 3]*horizon[:, 5]
                ]),
                'angulars': (4, [180/np.pi * horizon[:, 2], 180/np.pi * horizon[:, 5], 180/np.pi * accels[:, 2]]),
                'controls': (7, [-horizon[:, 6], horizon[:, 7]]),
            }

            # Update and draw the odometry/info graph
            for first_i, values_list in graph_data.values():
                for j, values in enumerate(values_list):
                    self.y[first_i+j, -(config.mpc_N+1):] = values
                    self.plot_items[first_i+j].setData(self.x[first_i+j], self.y[first_i+j])

        except Empty:
            pass
        except Exception as e:
            print("Graph error: ", traceback.print_exception(e))
            self.close()


