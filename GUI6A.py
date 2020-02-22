from PyQt5.QtGui import *  
from PyQt5.QtGui import QValidator
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication

###############################################################################
#Import Needed Machine Learning Functions
import numpy as np
#Need 2 Functions to extract all the data from csv files
from ExtractDataFunctions import ExtractAccelerometerData
from ExtractDataFunctions import ExtractAcousticData

#Need 3 Functions to Organize all the all the data
from OrganizationFunctions import Inputs2CondensedForm
from OrganizationFunctions import System2CondensedForm
from OrganizationFunctions import AllData2WorkingForm

#Need 1 Function to Organize Files into TestDataFrame
from FeatureFunctions import getTESTDataFrame

#Need 2 Functions for Graphing
from FeatureFunctions import getGraphs
from FeatureFunctions import truncate

#Need # Functions to perform Machine Learning
from MachineLearningFunctions import getTESTMatrix
from MachineLearningFunctions import GetTrainingData
from MachineLearningFunctions import TrainModel
from MachineLearningFunctions import PredictModel
from MachineLearningFunctions import PredictProbModel
###############################################################################

import os
import sip
import csv


from PyQt5.uic import loadUiType
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
message = ""
"""
These RegExp are used to check for user input correctness

"""
regexp_checkCSV = QRegExp('^\/([A-z0-9-_+\s]+\/)*([A-z0-9]+\.(csv))$')
validator = QRegExpValidator(regexp_checkCSV)

regexp_checkint = QIntValidator()
intvalidator = QRegExpValidator(regexp_checkint)

regexp_checkdouble = QDoubleValidator()
doublevalidator = QRegExpValidator(regexp_checkdouble)
	
Ui_MainWindow, QMainWindow = loadUiType('MQAAdraft4_3.1.ui')

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ): #provide values for attributes at runtime
        super(Main, self).__init__()
        self.setupUi(self)
        self.pushBrowse.clicked.connect(self.selectFile)
        self.pushBrowse_3.clicked.connect(self.selectFile2)
        self.pushBrowse_2.clicked.connect(self.selectmlFile)
        self.pushApply.clicked.connect(self.apply)
        self.pushRun.clicked.connect(self.run)
        self.HomeDirectory = os.getcwd() #saves the primary working directory
        self.directory = os.listdir(self.HomeDirectory)
        self.saveBttn.clicked.connect(self.file_save)
        self.actionOpen.triggered.connect(self.file_open)
        self.actionReset.triggered.connect(self.Reset)
        self.message = 0
        #self.UI = []
        self.reset = 0
        
        
        #check Accelerometer Data input for correct .csv file
        self.inputFile.setValidator(validator)
        self.inputFile.textChanged.connect(self.check_state)
        self.inputFile.textChanged.emit(self.inputFile.text())
        #check Acoustic Data input for correct .csv file
        self.inputFile2.setValidator(validator)
        self.inputFile2.textChanged.connect(self.check_state)
        self.inputFile2.textChanged.emit(self.inputFile2.text())
        #check User input for correct .csv file for ml
        self.mlData.setValidator(validator)
        self.mlData.textChanged.connect(self.check_state)
        self.mlData.textChanged.emit(self.mlData.text())
        #horsepower
        self.horsepower.setValidator(regexp_checkdouble)
        self.horsepower.textChanged.connect(self.check_state)
        self.horsepower.textChanged.emit(self.horsepower.text())
        #rated voltage
        self.voltage.setValidator(regexp_checkdouble)
        self.voltage.textChanged.connect(self.check_state)
        self.voltage.textChanged.emit(self.voltage.text())
        #phase
        self.phase.setValidator(regexp_checkdouble)
        self.phase.textChanged.connect(self.check_state)
        self.phase.textChanged.emit(self.phase.text())
        #shaft speed
        self.shaftSpeed.setValidator(regexp_checkdouble)
        self.shaftSpeed.textChanged.connect(self.check_state)
        self.shaftSpeed.textChanged.emit(self.shaftSpeed.text())
        #Num of rolling elements
        self.numberofElements.setValidator(regexp_checkint)
        self.numberofElements.textChanged.connect(self.check_state)
        self.numberofElements.textChanged.emit(self.numberofElements.text())
        #diameter of rolling elements
        self.diameterofElements.setValidator(regexp_checkdouble)
        self.diameterofElements.textChanged.connect(self.check_state)
        self.diameterofElements.textChanged.emit(self.diameterofElements.text())
        #pitch diameter
        self.pitchDiameter.setValidator(regexp_checkdouble)
        self.pitchDiameter.textChanged.connect(self.check_state)
        self.pitchDiameter.textChanged.emit(self.pitchDiameter.text())
        #Contact angle
        self.contactAngle.setValidator(regexp_checkdouble)
        self.contactAngle.textChanged.connect(self.check_state)
        self.contactAngle.textChanged.emit(self.contactAngle.text())
        #Frequency Acoustic
        self.samFreqAcst.setValidator(regexp_checkdouble)
        self.samFreqAcst.textChanged.connect(self.check_state)
        self.samFreqAcst.textChanged.emit(self.samFreqAcst.text())
        #Frequency Accelerometer
        self.samFreqAccl.setValidator(regexp_checkdouble)
        self.samFreqAccl.textChanged.connect(self.check_state)
        self.samFreqAccl.textChanged.emit(self.samFreqAccl.text())
        
        
        
   
    def check_state(self, *args, **kwargs): 
        #this function is changes the color of the lineedit fields depending on state
        sender = self.sender()
        validator = sender.validator()
        state = validator.validate(sender.text(), 0)[0]
        if state == QValidator.Acceptable:
            color = '#c4df9b' # green
        elif state == QtGui.QValidator.Intermediate:
            color = '#fff79a' # yellow
        else:
            color = '#f6989d' # red
        sender.setStyleSheet('QLineEdit { background-color: %s }' % color)
        
    #creates a dictionary from the saved CSV 
    def file_open(self): 
        #function called when the open file action in triggered. Creates a dictionary from a CSV file.
        filename = QFileDialog.getOpenFileName()[0]
        reader = csv.reader(open(filename, 'r'))
        d = {}
        for row in reader:
            k, v = row
            d[k] = v
                   
        print(d)
        
        self.setTextInfile(d)
        return True
    
 ############################################################################################  
    #used the dicitonary created above to assign saved variables to input parameters
    def setTextInfile(self, d):
        self.inputName.setText(d['inputName'])
        self.inputApplication.setText(d['inputApplication'])
        self.inputModelnum.setText(d['inputModelnum'])
        self.inputSavingalias.setText(d['inputSavingalias'])
        self.inputFile.setText(d['inputFile'])
        self.mlData.setText(d['mlData'])
        self.horsepower.setText(d['horsepower'])
        self.voltage.setText(d['voltage'])
        self.phase.setText(d['phase'])
        self.shaftnum.setText(d['shaftnum'])
        self.shaftSpeed.setText(d['shaftSpeed'])
        self.numberofElements.setText(d['numberofElements'])
        self.diameterofElements.setText(d['diameterofElements'])
        self.pitchDiameter.setText(d['pitchDiameter'])
        self.contactAngle.setText(d['contactAngle'])
        self.samFreqAcst.setText(d['samFreq'])
        
 ############################################################################################  
    
    """
    Hmm i wonder if this is used to save the file?
    """
    """
    NEEDS UPDATED VARIABLE NAMES
    """
    def file_save(self,): 
        #called when the save btn is clicked. converts user input to dictionary
        #then to dataframe then to csv file. 
        dict = CreateSaveDictionary(self.inputName.text(),\
                                    self.inputApplication.text(),\
                                    self.inputModelnum.text(),\
                                    self.inputSavingalias.text(),\
                                    self.inputFile.text(),\
                                    self.mlData.text(), \
                                    self.horsepower.text(), \
                                    self.voltage.text(), \
                                    self.phase.text(), \
                                    self.shaftnum.text(),\
                                    self.shaftSpeed.text(),\
                                    self.numberofElements.text(), \
                                    self.diameterofElements.text(), \
                                    self.pitchDiameter.text(), \
                                    self.contactAngle.text(), \
                                    self.samFreqAcst.text())
        CreateCSVfromDict(dict)
   
 ############################################################################################  
       
    #clearly selects file
    def selectFile(self,):
        self.inputFile.setText(QFileDialog.getOpenFileName()[0])
    #clearly selects file
    def selectFile2(self,):
        self.inputFile2.setText(QFileDialog.getOpenFileName()[0])
    #selects file but for a ml data   
    def selectmlFile(self,):
        self.mlData.setText(QFileDialog.getOpenFileName()[0])
     
        
############################################################################################         
    """
    Apply checks user inputs and then assigns them to function parameter variables
    In case the user doesn't supply input for a specific field, default inputs will
    be inserted.
    """   
    def apply(self,):
        
        MissingField = None
        
        # Graphic-Name/ID               Name -(inputName) 
        if self.inputName.text() != "":
            self.Name_ID = self.inputName.text()
        else:
            MissingField = "Name/ID"
        
        # Graphic-Application           Name -(inputApplication)
        if self.inputApplication.text() != "":
            self.Application = self.inputApplication.text()
        else:
            MissingField = "Application"
        
        # Graphic-Model #               Name -(inputModelnum)
        if self.inputModelnum.text() != "":
                self.ModelNumber = self.inputModelnum.text()
        else:
            MissingField = "Model #"
        
        # Graphic-Saving Alias          Name -(inputSavingAlias)
        if self.inputSavingalias.text() != "":
           self.SavingAlias = self.inputSavingalias.text()+".csv"
           self.inputSavingalias.setText(self.SavingAlias)
           SavingAlias = self.inputSavingalias.text()
        else:
            MissingField = "Saving Alias"   
            
        # Graphic-Accel Data            Name -(inputFile)
        if self.inputFile.text() != "":
            self.AccelerometerDataFilename =  self.inputFile.text()
        else:
            MissingField = "Accel Data"
        
        # Graphic-Acst Data            Name -(inputFile2)
        if self.inputFile2.text() != "":
            self.AcousticDataFilename =  self.inputFile2.text()
        else:
            MissingField = "Acst Data"
            
        # Graphic-ML Data              Name -(mlData)
        if self.mlData.text() != "":
            self.MLDataFilename = self.mlData.text() 
        else:
            MissingField = "ML Data"
        
        # Graphic-horsepower           Name -(horsepower)
        if self.horsepower.text() != "":
            self.Horsepower = float(self.horsepower.text())
        else:
            MissingField = "horsepower"
            
        # Graphic-Rated Voltage         Name -(voltage)
        if self.voltage.text() != "":
            self.RatedVoltage = float(self.voltage.text())
        else:
            MissingField = "Rated Voltage"
            
        # Graphic-Phase                 Name -(phase)
        if self.phase.text() != "":
            self.Phase = self.phase.text()
        else:
            MissingField = "Phase"
            
        # Graphic- # Shafts             Name -(shaftnum)
        if self.shaftnum.text() != "":
            self.NumberofShafts = self.shaftnum.text()
        else:
            MissingField = "# Shafts"
            
        # Graphic- Type                 Name -(ACDC)
        if self.ACDC.currentText() != "":
            self.ACorDC = self.ACDC.currentText()
        else:
            MissingField = "Type: "
            
        # Graphic-Pole Pairs            Name -(numberPoles)
        if self.numberPoles.text() != "":
            self.NumberOfPolePairs = float(self.numberPoles.text())
        else:
            MissingField = "Pole Pairs"
         
        # Graphic-Shaft Speed           Name -(shaftSpeed)
        if self.shaftSpeed.text() != "":
            self.ShaftSpeed = float(self.shaftSpeed.text())
        else:
            MissingField = "Shaft Speed"
            
        # Graphic-No. Of Rolling Elements           Name -(numberofElements)
        if self.numberofElements.text() != "":
            self.NumberOfRollingElements = int(self.numberofElements.text())
        else:
            MissingField = "No. Of Rolling Elements"
            
        # Graphic-Diameter of Rolling Elements      Name -(numberPoles)
        if self.diameterofElements.text() != "":
            self.DiameterOfRollingElements = float(self.diameterofElements.text())   
        else:
            MissingField = "Diameter of Rolling Elements"
            
        # Graphic-Pitch Diameter        Name -(diameterofElements)
        if self.pitchDiameter.text() != "":
            self.PitchDiameter = float(self.pitchDiameter.text())
        else:
            MissingField = "Pitch Diameter"
        
        # Graphic-Contact Angle         Name -(contactAngle)
        if self.contactAngle.text() != "":
            self.ContactAngle = float(self.contactAngle.text())
        else:
            MissingField = "Contact Angle"
            
        # Graphic-Accl Freq             Name -(samFreqAccl)
        if self.samFreqAccl.text() != "":
            self.AccelerometerSamplingFrequency = float(self.samFreqAccl.text())
        else:
            MissingField = "Accl Freq" 
            
        # Graphic-Acst Freq             Name -(samFreqAcst)
        if self.samFreqAcst.text() != "":
            self.AcousticSamplingFrequency = float(self.samFreqAcst.text()) 
        else:
            MissingField = "Acst Freq"
    
         
        #Popup
        if MissingField == None:
            MissingField = "Applied Successfully\nTest"
        self.popup = MyPopup(MissingField)
        self.popup.setGeometry(QtCore.QRect(100, 100, 400, 200))
        self.popup.show()
 
 ############################################################################################ 
    
    def addgraph(self,WhichGraph,WhichPlot,WhichCanvas, WhichToolbar, WhichLocation, WhichFig):
        WhichCanvas = FigureCanvas(WhichFig)
        WhichGraph.addWidget(WhichCanvas)
        WhichToolbar = NavigationToolbar(WhichCanvas,WhichLocation, coordinates=True)
        WhichGraph.addWidget(WhichToolbar)
        
        return WhichCanvas, WhichToolbar
       
 ############################################################################################        
    def getPlot(self,WhichGraph,WhichPlot,WhichCanvas, WhichToolbar, WhichLocation, WhichFig, plotinfo):
        X = plotinfo[0]
        Y = plotinfo[1]
        xlabel = plotinfo[2]
        ylabel = plotinfo[3]
        Title = plotinfo[4]
        if self.reset != 0:
            WhichPlot.cla()
            
        WhichPlot.plot(X,Y,c = np.random.rand(3,))
        WhichPlot.set_xlabel(xlabel, fontsize=12)
        WhichPlot.set_ylabel(ylabel, fontsize=12)
        WhichPlot.set_title(Title)
        WhichFig.set_tight_layout(True)
        WhichPlot.grid(True)
        if self.reset != 0: 
            WhichCanvas.draw()
        
    
        return WhichCanvas, WhichToolbar
 
 ############################################################################################    
    
    def CloseWhichGraph(self,WhichGraph,WhichPlot,WhichCanvas, WhichToolbar, WhichLocation, WhichFig):
        WhichGraph.removeWidget(WhichCanvas)
        WhichCanvas.close()
        WhichCanvas.draw()
        WhichGraph.removeWidget(WhichToolbar)
        WhichToolbar.close()
        
        return WhichCanvas, WhichToolbar
    
############################################################################################ 
    
    def run(self,): #called when run is clicked
        
        if self.reset == 0:
            #instantiate the figures
            self.fig0 = plt.figure()
            self.sub0 = self.fig0.subplots()
            self.canvas01 = None
            self.toolbar01 = None
            
            self.fig1 = plt.figure() 
            self.sub1 = self.fig1.subplots()
            self.canvas02 = None
            self.toolbar02 = None
            
            self.fig2 = plt.figure()
            self.sub2 = self.fig2.subplots()
            self.canvas11 = None
            self.toolbar11 = None
            
            self.fig3 = plt.figure()
            self.sub3 = self.fig3.subplots()
            self.canvas12 = None
            self.toolbar12 = None
            
            self.fig4 = plt.figure()
            self.sub4 = self.fig4.subplots()
            self.canvas21 = None
            self.toolbar21 = None
            
            self.fig5 = plt.figure()
            self.sub5 = self.fig5.subplots()
            self.canvas01_2 = None
            self.toolbar01_2 = None
            
            self.fig6 = plt.figure() 
            self.sub6 = self.fig6.subplots()
            self.canvas02_2 = None
            self.toolbar02_2 = None
            
            self.fig7 = plt.figure()
            self.sub7 = self.fig7.subplots()
            self.canvas11_2 = None
            self.toolbar11_2 = None
            
            self.fig8 = plt.figure()
            self.sub8 = self.fig8.subplots()
            self.canvas12_2 = None
            self.toolbar12_2 = None
            
            self.fig9 = plt.figure()
            self.sub9 = self.fig9.subplots()
            self.canvas21_2 = None
            self.toolbar21_2 = None
            
        #--------------------------------------------------------------------#
        #begin calling ml functions for processing

        #System/Sensor Known Constants
        A2DResolution = 16
        VoltageMax = 5
        VoltageMin = 0
        SensorRunTime = 3
        AccelerationMax = 50 
        AccelerationMin = -50
                
        #Convert User Inputs into a condensed form
        OnlyUserInput = Inputs2CondensedForm(self.Name_ID, self.Application, self.ModelNumber, self.SavingAlias,\
                                             self.AccelerometerDataFilename, self.AcousticDataFilename, \
                                             self.MLDataFilename, self.Horsepower, self.RatedVoltage, self.ACorDC, \
                                             self.NumberOfPolePairs, self.NumberofShafts, \
                                             self.ShaftSpeed, self.NumberOfRollingElements, \
                                             self.DiameterOfRollingElements,self.PitchDiameter, self.ContactAngle, \
                                             self.AccelerometerSamplingFrequency, self.AcousticSamplingFrequency)
        
        SystemInputs = System2CondensedForm(A2DResolution,VoltageMax,VoltageMin,SensorRunTime,AccelerationMax,AccelerationMin)
        
        #Acquire Accelerometer Actual Data
        time, amp, Voltage, Acceleration = ExtractAccelerometerData(OnlyUserInput,SystemInputs)

        #Acquire Acoustic Actual Data
        Channel1Time,Channel1Value = ExtractAcousticData(OnlyUserInput,SystemInputs)    
        
        #Put All Data into Working Form
        trial = 2 #Select the instance of the data
        AllData = AllData2WorkingForm(OnlyUserInput,SystemInputs,time[trial], \
                                      amp[trial], Voltage[trial], Acceleration[trial],\
                                     Channel1Time,Channel1Value)
        
        #Machine Learning
        TestDF = getTESTDataFrame(AllData)
        TestMatrix = getTESTMatrix(TestDF)
        Xall_train, Yall_train, dataset = GetTrainingData(AllData)
        FinalClassifier = TrainModel(Xall_train, Yall_train)
        
        #Predict
        prediction,prediction_string = PredictModel(FinalClassifier,TestMatrix)
        prediction_proba = PredictProbModel(FinalClassifier,TestMatrix)

        #End of ml functions
        #--------------------------------------------------------------------#
        plt.close('all')
        plot0info,plot2info,plot1info,plot3info,plot4info,\
        plot5info,plot7info,plot6info,plot8info,plot9info = getGraphs(AllData)
        
        
        #ACCELEROMETER PLOTTING
        self.canvas01, self.toolbar01 = self.getPlot(self.graph01UI, self.sub0,self.canvas01, self.toolbar01,self.graph01,self.fig0,plot0info)
        self.canvas02, self.toolbar02 = self.getPlot(self.graph02UI, self.sub1,self.canvas02, self.toolbar02,self.graph02,self.fig1,plot1info)
        self.canvas11, self.toolbar11 = self.getPlot(self.graph11UI, self.sub2,self.canvas11, self.toolbar11,self.graph11,self.fig2,plot2info)
        self.canvas12, self.toolbar12 = self.getPlot(self.graph12UI, self.sub3,self.canvas12, self.toolbar12,self.graph12,self.fig3,plot3info)
        self.canvas21, self.toolbar21 = self.getPlot(self.graph21UI, self.sub4,self.canvas21, self.toolbar21,self.graph21,self.fig4,plot4info)
        
        
        #ACOUSTIC PLOTTING
        self.canvas01_2, self.toolbar01_2 = self.getPlot(self.graph01UI_2, self.sub5,self.canvas01_2, self.toolbar01_2,self.graph01_2,self.fig5,plot5info)
        self.canvas02_2, self.toolbar02_2 = self.getPlot(self.graph02UI_2, self.sub6,self.canvas02_2, self.toolbar02_2,self.graph02_2,self.fig6,plot6info)
        self.canvas11_2, self.toolbar11_2 = self.getPlot(self.graph11UI_2, self.sub7,self.canvas11_2, self.toolbar11_2,self.graph11_2,self.fig7,plot7info)
        self.canvas12_2, self.toolbar12_2 = self.getPlot(self.graph12UI_2, self.sub8,self.canvas12_2, self.toolbar12_2,self.graph12_2,self.fig8,plot8info)
        self.canvas21_2, self.toolbar21_2 = self.getPlot(self.graph21UI_2, self.sub9,self.canvas21_2, self.toolbar21_2,self.graph21_2,self.fig9,plot9info)
    
        if self.reset == 0:
            self.canvas01, self.toolbar01 = self.addgraph(self.graph01UI, self.sub0,self.canvas01, self.toolbar01,self.graph01,self.fig0)
            self.canvas02, self.toolbar02 = self.addgraph(self.graph02UI, self.sub1,self.canvas02, self.toolbar02,self.graph02,self.fig1)
            self.canvas11, self.toolbar11 = self.addgraph(self.graph11UI, self.sub2,self.canvas11, self.toolbar11,self.graph11,self.fig2)
            self.canvas12, self.toolbar12 = self.addgraph(self.graph12UI, self.sub3,self.canvas12, self.toolbar12,self.graph12,self.fig3)
            self.canvas21, self.toolbar21 = self.addgraph(self.graph21UI, self.sub4,self.canvas21, self.toolbar21,self.graph21,self.fig4)
            self.canvas01_2, self.toolbar01_2 = self.addgraph(self.graph01UI_2, self.sub5,self.canvas01_2, self.toolbar01_2,self.graph01_2,self.fig5)
            self.canvas02_2, self.toolbar02_2 = self.addgraph(self.graph02UI_2, self.sub6,self.canvas02_2, self.toolbar02_2,self.graph02_2,self.fig6)
            self.canvas11_2, self.toolbar11_2 = self.addgraph(self.graph11UI_2, self.sub7,self.canvas11_2, self.toolbar11_2,self.graph11_2,self.fig7)
            self.canvas12_2, self.toolbar12_2 = self.addgraph(self.graph12UI_2, self.sub8,self.canvas12_2, self.toolbar12_2,self.graph12_2,self.fig8)
            self.canvas21_2, self.toolbar21_2 = self.addgraph(self.graph21UI_2, self.sub9,self.canvas21_2, self.toolbar21_2,self.graph21_2,self.fig9)

            
            
        self.BSF.setText(str(truncate(TestDF["BSF"].values[0],2)))
        self.BPFI.setText(str(truncate(TestDF["BPFI"].values[0],2)))
        self.BPFO.setText(str(truncate(TestDF["BPFO"].values[0],2)))
        self.FTF.setText(str(truncate(TestDF["FTF"].values[0],2)))
        
        self.earlyEdit.setText(str(truncate(prediction_proba[0,0],2)))
        self.suspectEdit.setText(str(truncate(prediction_proba[0,1],2)))
        self.normalEdit.setText(str(truncate(prediction_proba[0,2],2)))

        self.reset = 1 
 
############################################################################################         
    def close_application(self,): #self explanitory
        sys.exit()
       
 ############################################################################################   
    #Associated with file->reset
    def Reset(self,):
        
        if self.reset != 0:
        
            self.canvas01, self.toolbar01 = self.CloseWhichGraph(self.graph01UI, self.sub0,self.canvas01, self.toolbar01,self.graph01,self.fig0)
            self.canvas02, self.toolbar02 = self.CloseWhichGraph(self.graph02UI, self.sub1,self.canvas02, self.toolbar02,self.graph02,self.fig1)
            self.canvas11, self.toolbar11 = self.CloseWhichGraph(self.graph11UI, self.sub2,self.canvas11, self.toolbar11,self.graph11,self.fig2)
            self.canvas12, self.toolbar12 = self.CloseWhichGraph(self.graph12UI, self.sub3,self.canvas12, self.toolbar12,self.graph12,self.fig3)
            self.canvas21, self.toolbar21 = self.CloseWhichGraph(self.graph21UI, self.sub4,self.canvas21, self.toolbar21,self.graph21,self.fig4)
            self.canvas01_2, self.toolbar01_2 = self.CloseWhichGraph(self.graph01UI_2, self.sub5,self.canvas01_2, self.toolbar01_2,self.graph01_2,self.fig5)
            self.canvas02_2, self.toolbar02_2 = self.CloseWhichGraph(self.graph02UI_2, self.sub6,self.canvas02_2, self.toolbar02_2,self.graph02_2,self.fig6)
            self.canvas11_2, self.toolbar11_2 = self.CloseWhichGraph(self.graph11UI_2, self.sub7,self.canvas11_2, self.toolbar11_2,self.graph11_2,self.fig7)
            self.canvas12_2, self.toolbar12_2 = self.CloseWhichGraph(self.graph12UI_2, self.sub8,self.canvas12_2, self.toolbar12_2,self.graph12_2,self.fig8)
            self.canvas21_2, self.toolbar21_2 = self.CloseWhichGraph(self.graph21UI_2, self.sub9,self.canvas21_2, self.toolbar21_2,self.graph21_2,self.fig9)
  
    
            self.reset = 0
        
############################################################################################ 

class MyPopup(QWidget): #creates popup windows
    def __init__(self, message):
        QWidget.__init__(self)
        alertholder = QLabel(self)
        alertholder.setText(message)
        alertholder.setAlignment(Qt.AlignCenter)
        
        vbox = QVBoxLayout()
        vbox.addWidget(alertholder)
        self.setLayout(vbox)
 
        
############################################################################################        
if __name__ == "__main__": #instantiates GUI and opens it
    from PyQt5 import *
    import sys
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    app.exec_()

    