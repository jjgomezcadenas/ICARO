import sys
import numpy as np
import csv
from PyQt5 import QtCore, QtWidgets, uic
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from PyQt5.uic import loadUiType


# # My functions
# from MP_wrappers import BLR_lambda
# from MP_wrappers import BLR_batch
# from cal_IO_lib import read_CALHF
# from DBLR_lab import find_baseline
# from fit_library import gauss
# import fit_library as fit
#
#
qtCreatorFile = "tutorial.ui" # Enter file here.
Ui_MainWindow, QMainWindow = loadUiType(qtCreatorFile)

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}

        self.mplfigs.itemClicked.connect(self.changefig)

        fig = Figure()
        self.addmpl(fig)

    def addfig(self, name, fig):
        self.fig_dict[name] = fig
        self.mplfigs.addItem(name)

    def changefig(self, item):
        text = item.text()
        self.rmmpl()
        self.addmpl(self.fig_dict[text])

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas,
                self.mplwindow, coordinates=True)
        self.mplvl.addWidget(self.toolbar)
    # def addmpl(self, fig):
    #     self.canvas = FigureCanvas(fig)
    #     self.mplvl.addWidget(self.canvas)
    #     self.canvas.draw()
    #     self.toolbar = NavigationToolbar(self.canvas,
    #             self, coordinates=True)
    #     self.addToolBar(self.toolbar)
    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()



if __name__ == '__main__':
    import sys
    #from PyQt5 import QtGui
    import numpy as np

    fig1 = Figure()
    ax1f1 = fig1.add_subplot(111)
    ax1f1.plot(np.random.rand(5))

    fig2 = Figure()
    ax1f2 = fig2.add_subplot(121)
    ax1f2.plot(np.random.rand(5))
    ax2f2 = fig2.add_subplot(122)
    ax2f2.plot(np.random.rand(10))

    fig3 = Figure()
    ax1f3 = fig3.add_subplot(111)
    ax1f3.pcolormesh(np.random.rand(20,20))

    app = QtWidgets.QApplication(sys.argv)
    #app = QtGui.QApplication(sys.argv)
    main = Main()
    main.addfig('One plot', fig1)
    main.addfig('Two plots', fig2)
    main.addfig('Pcolormesh', fig3)
    main.show()
    sys.exit(app.exec_())

#
# Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
#
#
# class LAB_data():
#     fig1 = Figure()
#     fig2 = Figure()
#     axes={'ax1':0,'ax2':0,'ax3':0}
#     # Axes 1 & 2 for figure 1. Axes 3 for figure 2
#     baseline_txt = fig1.text(0.6,0.8, ('BASELINE = '))
#     mu_txt = fig2.text(0.15,0.8, ('MU = '))
#     sigma_txt = fig2.text(0.15,0.75, ('SIGMA = '))
#     res_txt = fig2.text(0.15,0.7, ('RES(fwhm) = '))
#
#     def_path = "F:/DATOS_DAC/CALIBRATION_FEB_2017_NB/"
#
#     files={'fname':"F:/DATOS_DAC/CALIBRATION_FEB_2017_NB/box0_100u.h5",
#           'conf_name':"../COEFF.conf"}
#
#
#     PMT_data = {'PMT_n':0, 'event_n':0, 'blr':0.8E-3, 'cf':1E-6,
#                   'thr':5, 'spe':20.0, 'accum_floor':3500, 'i':1, 'thre':5}
#
#     flags = {'config':False}
#
#     hist = {'bins':50, 'Low_limit':10, 'High_limit':1000000, 'n_events':10}
#
#     FECal_DATA = np.ones((24,3),dtype=float)
#     FECal_DATA[:,1]=FECal_DATA[:,1]*PMT_data['blr']
#     FECal_DATA[:,2]=FECal_DATA[:,2]*PMT_data['cf']
#
#     energy_raw = []
#
#
# class CORE():
#     def __init__(self,upper_class):
#         self.uc = upper_class
#
#     def f_DBLR(self):
#
#         f = read_CALHF(self.uc.d.files['fname'], int(self.uc.spinBox_pmt.value()),
#                                    int(self.uc.spinBox_event.value()))
#
#         output_BLR = BLR_lambda( f,
#                                  coef = self.uc.d.PMT_data['blr'],
#                                  thr = self.uc.d.PMT_data['thr'],
#                                  acum_FLOOR = self.uc.d.PMT_data['accum_floor'],
#                                  coef_clean = self.uc.d.PMT_data['cf'],
#                                  filter=self.uc.checkBox_filt.isChecked(),
#                                  i_factor = self.uc.d.PMT_data['i'],
#                                  e_thr = self.uc.d.PMT_data['thre'],
#                                  SPE = self.uc.d.PMT_data['spe'] )
#
#         print output_BLR['ENERGY']
#
#         self.uc.lcdNumber_2.display(np.std(output_BLR['recons'][100:1000],ddof=1))
#         self.uc.lcdNumber.display(output_BLR['ENERGY'])
#         self.uc.start_t.setText(str(output_BLR['LIMIT_L']))
#         self.uc.end_t.setText(str(output_BLR['LIMIT_H']))
#
#
#         self.uc.d.axes['ax1'].plot(output_BLR['recons'])
#
#         if (self.uc.checkBox_accu.isChecked() == True):
#             self.uc.d.axes['ax2'].plot(output_BLR['acum'])
#         if (self.uc.checkBox_pre.isChecked() == True):
#             self.uc.d.axes['ax1'].plot(output_BLR['signal_daq'])
#
#         self.uc.d.baseline_txt.remove()
#         self.uc.d.baseline_txt=self.uc.d.fig1.text(0.6,0.8, ('BASELINE = %0.2f'
#                                           % (find_baseline(f[:1400]))))
#         self.uc.canvas1.draw()
#
#
#     def f_RES(self):
#
#         self.uc.d.energy_raw = BLR_batch(self.uc.d.files['fname'],
#                                     coef = self.uc.d.PMT_data['blr'],
#                                     thr = self.uc.d.PMT_data['thr'],
#                                     acum_FLOOR = self.uc.d.PMT_data['accum_floor'],
#                                     coef_clean = self.uc.d.PMT_data['cf'],
#                                     SPE = self.uc.d.PMT_data['spe'],
#                                     e_thr = self.uc.d.PMT_data['thre'],
#                                     filter = self.uc.checkBox_filt.isChecked(),
#                                     i_factor = self.uc.d.PMT_data['i'],
#                                     point = self.uc.d.PMT_data['PMT_n'],
#                                     n_events = self.uc.d.hist['n_events']
#                                     )
#
#         self.uc.redraw_b.clicked.emit()
#         #
#         # #Outlayers Filtering
#         # LIMITL_hist = float(self.uc.Low_limit_t.text())
#         # LIMITH_hist = float(self.uc.High_limit_t.text())
#         #
#         # condition = (self.uc.d.energy_raw>LIMITL_hist)*(self.uc.d.energy_raw<LIMITH_hist)
#         # energy = np.extract(condition,self.uc.d.energy_raw)
#         #
#         # print energy
#         #
#         # try:
#         #     coeff_fit, perr, hist, bin_centres = fit.gauss1_fit(energy,
#         #                                                     '','','',
#         #                                                     self.uc.d.hist['bins'],1,0)
#         #     hist_fit = fit.gauss(bin_centres, coeff_fit[0],coeff_fit[1],coeff_fit[2])
#         #     res = np.abs(coeff_fit[2])*2.35/coeff_fit[1]*100.0
#         #     err_res =  2.35*100.0*(np.abs(perr[2]/coeff_fit[1])+np.abs(perr[1]*coeff_fit[2])/np.square(coeff_fit[1]))
#         # except RuntimeError:
#         #     print "Fitting Problems"
#         #     hist_fit = 0
#         #     bin_centres = 0
#         #     err_res = 0
#         #
#         # self.uc.d.axes['ax3'].hist(energy, self.uc.d.hist['bins'], facecolor='green')
#         # self.uc.d.axes['ax3'].plot(bin_centres, hist_fit, 'r--', linewidth=1)
#         # self.uc.d.axes['ax3'].grid(True)
#         # #pinta2_ER.xlabel("Energy (pe)")
#         # #pinta_ER.ylabel("Hits")
#         #
#         # self.uc.d.fig2.suptitle("Energy Resolution")
#         #
#         # self.uc.d.mu_txt.remove()
#         # self.uc.d.sigma_txt.remove()
#         # self.uc.d.res_txt.remove()
#         # self.uc.d.mu_txt=self.uc.d.fig2.text(0.15,0.8, ('MU = %0.3f ( +/- %0.2f)' % (coeff_fit[1] , perr[1])))
#         # self.uc.d.sigma_txt=self.uc.d.fig2.text(0.15,0.75, ('SIGMA = %0.3f ( +/- %0.2f)' % (np.abs(coeff_fit[2]) , perr[2])))
#         # self.uc.d.res_txt=self.uc.d.fig2.text(0.15,0.70, ('RES(fwhm) = %0.2f (+/- %0.2f )(percent)' % (res, err_res)))
#         #
#         # self.uc.canvas2.draw()
#
#
# class Aux_Buttons():
#     def __init__(self,upper_class):
#         self.uc = upper_class
#
#     def f_quit(self):
#         QtCore.QCoreApplication.instance().quit()
#
#     def f_clear(self,option):
#
#         if (option==1):
#             self.uc.d.axes['ax1'].cla()
#             self.uc.d.axes['ax2'].cla()
#             for txt in self.uc.d.fig1.texts:
#                 txt.set_visible(False)
#             self.uc.canvas1.draw()
#         elif (option==2):
#             self.uc.d.axes['ax3'].cla()
#             for txt in self.uc.d.fig2.texts:
#                 txt.set_visible(False)
#             self.uc.canvas2.draw()
#
#
#     def f_CONFIG(self):
#         data_file = open(self.uc.d.files['conf_name'])
#         data_reader = csv.reader(data_file)
#
#         FECal_DATA_c = np.array(list(data_reader))
#         self.uc.d.FECal_DATA = FECal_DATA_c[1:,:].astype(np.float)
#
#         self.uc.blr_t.setText(str(self.uc.d.FECal_DATA[self.uc.spinBox_pmt.value(),1]))
#         self.uc.cf_t.setText(str(self.uc.d.FECal_DATA[self.uc.spinBox_pmt.value(),2]))
#
#         print self.uc.d.FECal_DATA
#         self.uc.d.flags['config'] = True
#
#     def f_coeff_update(self):
#         if (self.uc.d.flags['config'] == True):
#             self.uc.blr_t.setText(str(self.uc.d.FECal_DATA[self.uc.spinBox_pmt.value(),1]))
#             self.uc.cf_t.setText(str(self.uc.d.FECal_DATA[self.uc.spinBox_pmt.value(),2]))
#             self.uc.blr_t_2.setText(str(self.uc.d.FECal_DATA[self.uc.spinBox_pmt_2.value(),1]))
#             self.uc.cf_t_2.setText(str(self.uc.d.FECal_DATA[self.uc.spinBox_pmt_2.value(),2]))
#
#     def f_redraw(self):
#
#         #Outlayers Filtering
#         LIMITL_hist = float(self.uc.Low_limit_t.text())
#         LIMITH_hist = float(self.uc.High_limit_t.text())
#
#         condition = (self.uc.d.energy_raw>LIMITL_hist)*(self.uc.d.energy_raw<LIMITH_hist)
#         energy = np.extract(condition,self.uc.d.energy_raw)
#
#         print energy
#
#         try:
#             coeff_fit, perr, hist, bin_centres = fit.gauss1_fit(energy,
#                                                             '','','',
#                                                             self.uc.d.hist['bins'],1,0)
#             hist_fit = fit.gauss(bin_centres, coeff_fit[0],coeff_fit[1],coeff_fit[2])
#             res = np.abs(coeff_fit[2])*2.35/coeff_fit[1]*100.0
#             err_res =  2.35*100.0*(np.abs(perr[2]/coeff_fit[1])+np.abs(perr[1]*coeff_fit[2])/np.square(coeff_fit[1]))
#         except RuntimeError:
#             print "Fitting Problems"
#             hist_fit = 0
#             bin_centres = 0
#             err_res = 0
#
#         self.uc.d.axes['ax3'].hist(energy, self.uc.d.hist['bins'], facecolor='green')
#         self.uc.d.axes['ax3'].plot(bin_centres, hist_fit, 'r--', linewidth=1)
#         self.uc.d.axes['ax3'].grid(True)
#         #pinta2_ER.xlabel("Energy (pe)")
#         #pinta_ER.ylabel("Hits")
#
#         self.uc.d.fig2.suptitle("Energy Resolution")
#
#         self.uc.d.mu_txt.remove()
#         self.uc.d.sigma_txt.remove()
#         self.uc.d.res_txt.remove()
#         self.uc.d.mu_txt=self.uc.d.fig2.text(0.15,0.8, ('MU = %0.3f ( +/- %0.2f)' % (coeff_fit[1] , perr[1])))
#         self.uc.d.sigma_txt=self.uc.d.fig2.text(0.15,0.75, ('SIGMA = %0.3f ( +/- %0.2f)' % (np.abs(coeff_fit[2]) , perr[2])))
#         self.uc.d.res_txt=self.uc.d.fig2.text(0.15,0.70, ('RES(fwhm) = %0.2f (+/- %0.2f )(percent)' % (res, err_res)))
#
#         self.uc.canvas2.draw()
#
#
#     def float_v(self,number):
#         try:
#             return float(number)
#         except ValueError:
#             return 0.0
#
#     def int_v(self,number):
#         try:
#             return int(number)
#         except ValueError:
#             return 0
#
#     def store_data(self):
#         self.uc.d.PMT_data['PMT_n'] = self.int_v(self.uc.spinBox_pmt.value())
#         self.uc.d.PMT_data['blr'] = self.float_v(self.uc.blr_t.text())
#         self.uc.d.PMT_data['cf'] = self.float_v(self.uc.cf_t.text())
#         self.uc.d.PMT_data['thr'] = self.float_v(self.uc.thr_t.text())
#         self.uc.d.PMT_data['accum_floor'] = self.float_v(self.uc.accum_t.text())
#         self.uc.d.PMT_data['i'] = self.int_v(self.uc.spinBox_i.value())
#         self.uc.d.PMT_data['thre'] = self.float_v(self.uc.spinBox_thre.value())
#         self.uc.d.PMT_data['spe'] = self.float_v(self.uc.spe_t.text())
#         self.uc.d.hist['bins'] = self.int_v(self.uc.bins_t.text())
#         self.uc.d.hist['n_events'] = self.int_v(self.uc.events_t.text())
#         self.uc.d.hist['Low_limit'] = self.int_v(self.uc.Low_limit_t.text())
#         self.uc.d.hist['High_limit'] = self.int_v(self.uc.High_limit_t.text())
#
#     def mirror_DBLR2RES(self):
#         self.uc.spinBox_pmt_2.setValue(self.uc.spinBox_pmt.value())
#         self.uc.blr_t_2.setText(self.uc.blr_t.text())
#         self.uc.cf_t_2.setText(self.uc.cf_t.text())
#         self.uc.thr_t_2.setText(self.uc.thr_t.text())
#         self.uc.accum_t_2.setText(self.uc.accum_t.text())
#         self.uc.spinBox_i_2.setValue(self.uc.spinBox_i.value())
#         self.uc.spinBox_thre_2.setValue(self.uc.spinBox_thre.value())
#         self.uc.spe_t_2.setText(self.uc.spe_t.text())
#         self.store_data()
#
#     def mirror_RES2DBLR(self):
#         self.uc.spinBox_pmt.setValue(self.uc.spinBox_pmt_2.value())
#         self.uc.spinBox_thre.setValue(self.uc.spinBox_thre_2.value())
#         self.uc.spe_t.setText(self.uc.spe_t_2.text())
#         self.store_data()
#
#
# class Browsers():
#     def __init__(self,upper_class):
#         self.uc = upper_class
#
#     def path_browser(self):
#         file_aux = QtWidgets.QFileDialog.getOpenFileName(self.uc,
#                                         'Open file',
#                                         self.uc.d.def_path,
#                                         "Calibration Files H5 (*.h5)")
#
#         fname_aux = ([str(x) for x in file_aux])
#         self.uc.d.files['fname'] = fname_aux[0]
#         #Trick for Qstring converting to standard string
#         self.uc.file_path_t.setText(self.uc.d.files['fname'])
#
#
#     def conf_path_browser(self):
#         file_aux = QtWidgets.QFileDialog.getOpenFileName(self.uc,
#                                         'Open file',
#                                         './',
#                                         "Configuration File (*.conf)")
#
#         fname_aux = ([str(x) for x in file_aux])
#         self.uc.d.files['conf_name'] = fname_aux[0]
#         #Trick for Qstring converting to standard string
#         self.uc.conf_file_t.setText(self.uc.d.files['conf_name'])
#
#
#
# class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
#     def __init__(self):
#         QtWidgets.QMainWindow.__init__(self)
#         Ui_MainWindow.__init__(self)
#         self.setupUi(self)
#
#         #Data class
#         self.d = LAB_data()
#         self.c = CORE(self)
#         self.browser_tools = Browsers(self)
#         self.b_buttons = Aux_Buttons(self)
#         # Passes all the needed information to the constructor of the aux class
#
#         #Defaults
#         #DBLR
#         self.checkBox_filt.setChecked(True)
#         self.checkBox_accu.setChecked(False)
#         self.checkBox_pre.setChecked(False)
#         self.conf_file_t.setText(self.d.files['conf_name'])
#         self.file_path_t.setText(self.d.files['fname'])
#         self.blr_t.setText(str(self.d.PMT_data['blr']))
#         self.cf_t.setText(str(self.d.PMT_data['cf']))
#         self.thr_t.setText(str(self.d.PMT_data['thr']))
#         self.spe_t.setText(str(self.d.PMT_data['spe']))
#         self.accum_t.setText(str(self.d.PMT_data['accum_floor']))
#         self.spinBox_thre.setValue(self.d.PMT_data['thre'])
#         #Resolution
#         self.file_path_t_2.setText(self.d.files['fname'])
#         self.blr_t_2.setText(str(self.d.PMT_data['blr']))
#         self.cf_t_2.setText(str(self.d.PMT_data['cf']))
#         self.thr_t_2.setText(str(self.d.PMT_data['thr']))
#         self.accum_t_2.setText(str(self.d.PMT_data['accum_floor']))
#         self.spe_t_2.setText(str(self.d.PMT_data['spe']))
#         self.bins_t.setText(str(self.d.hist['bins']))
#         self.Low_limit_t.setText(str(self.d.hist['Low_limit']))
#         self.High_limit_t.setText(str(self.d.hist['High_limit']))
#         self.events_t.setText(str(self.d.hist['n_events']))
#         self.spinBox_thre_2.setValue(self.d.PMT_data['thre'])
#
#
#         #Button Calls
#         self.quit_b.clicked.connect(self.b_buttons.f_quit)
#         self.quit_b_2.clicked.connect(self.b_buttons.f_quit)
#         self.go_DBLR_b.clicked.connect(self.c.f_DBLR)
#         self.go_RES.clicked.connect(self.c.f_RES)
#         self.clear_b.clicked.connect(lambda: self.b_buttons.f_clear(1))
#         self.clear_b_2.clicked.connect(lambda: self.b_buttons.f_clear(2))
#
#         self.file_path_b.clicked.connect(self.browser_tools.path_browser)
#         self.conf_path_b.clicked.connect(self.browser_tools.conf_path_browser)
#         self.load_config_b.clicked.connect(self.b_buttons.f_CONFIG)
#         self.redraw_b.clicked.connect(self.b_buttons.f_redraw)
#
#         # Signals
#         self.spinBox_pmt.valueChanged.connect(self.b_buttons.f_coeff_update)
#         self.spinBox_pmt.valueChanged.connect(self.b_buttons.mirror_DBLR2RES)
#         self.spinBox_pmt_2.valueChanged.connect(self.b_buttons.f_coeff_update)
#         self.spinBox_pmt_2.valueChanged.connect(self.b_buttons.mirror_RES2DBLR)
#         self.spinBox_i.valueChanged.connect(self.b_buttons.mirror_DBLR2RES)
#         self.spinBox_thre_2.valueChanged.connect(self.b_buttons.mirror_RES2DBLR)
#         self.spinBox_thre.valueChanged.connect(self.b_buttons.mirror_DBLR2RES)
#
#         self.blr_t.editingFinished.connect(self.b_buttons.mirror_DBLR2RES)
#         self.cf_t.editingFinished.connect(self.b_buttons.mirror_DBLR2RES)
#         self.thr_t.editingFinished.connect(self.b_buttons.mirror_DBLR2RES)
#         self.accum_t.editingFinished.connect(self.b_buttons.mirror_DBLR2RES)
#         self.spe_t_2.editingFinished.connect(self.b_buttons.mirror_RES2DBLR)
#         self.spe_t.editingFinished.connect(self.b_buttons.mirror_DBLR2RES)
#         self.bins_t.editingFinished.connect(self.b_buttons.store_data)
#         self.events_t.editingFinished.connect(self.b_buttons.store_data)
#
#
#     def addmpl_1(self, fig):
#         # Matplotlib constructor
#         self.canvas1 = FigureCanvas(fig)
#         self.mpl_lay.addWidget(self.canvas1)
#         self.canvas1.draw()
#         self.toolbar = NavigationToolbar(self.canvas1, self.frame_plot,
#                                          coordinates=True)
#         self.mpl_lay.addWidget(self.toolbar)
#         self.d.axes['ax1'] = fig.add_subplot(111)
#         self.d.axes['ax2'] = self.d.axes['ax1'].twinx()
#
#
#     def addmpl_2(self, fig):
#         # Matplotlib constructor
#         self.canvas2 = FigureCanvas(fig)
#         self.mpl_lay2.addWidget(self.canvas2)
#         self.canvas2.draw()
#         self.toolbar = NavigationToolbar(self.canvas2, self.frame_plot,
#                                          coordinates=True)
#         self.mpl_lay2.addWidget(self.toolbar)
#         self.d.axes['ax3'] = fig.add_subplot(111)
#
#
# if __name__ == "__main__":
#     print('hello')
# #
#     app = QtWidgets.QApplication(sys.argv)
#     window = MyApp()
#     window.addmpl_1(window.d.fig1)
#     window.addmpl_2(window.d.fig2)
#     window.show()
#     sys.exit(app.exec_())
