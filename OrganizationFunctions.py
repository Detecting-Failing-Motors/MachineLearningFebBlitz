
# coding: utf-8

# In[1]:


#Import Native Libraries
import copy


# In[2]:


def Inputs2CondensedForm(Name_ID = None, Application = None, ModelNumber = None,                          SavingAlias = None, AccelerometerDataFilename = None,                          AcousticDataFilename = None, MLDataFilename = None,                          Horsepower = None, RatedVoltage = None, ACorDC = None,                          NumberOfPolePairs = None, NumberofShafts = None,                          ShaftSpeed = None, NumberOfRollingElements = None, DiameterOfRollingElements = None,                          PitchDiameter = None, ContactAngle = None,                          AccelerometerSamplingFrequency = None, AcousticSamplingFrequency = None):
    
    """
    Returns a dictionary of all user inputted information
    
    Inputs2CondensedForm(
        Name_ID
        Application -
        ModelNumber -
        SavingAlias - 
        AccelerometerDataFilename -
        AcousticDataFilename - 
        MLDataFilename - 
        Horsepower - 
        RatedVoltage - 
        ACorDC -
        NumberOfPolePairs - 
        NumberofShafts -
        ShaftSpeed - #Shaft rotational speed [Hz], n
        NumberOfRollingElements - No. of rolling elements [-], N
        DiameterOfRollingElements - #Diameter of a rolling element [mm], Bd
        PitchDiameter - #Pitch diameter [mm], Pd
        ContactAngle - #Contact angle [rad], Phi
        AccelerometerSamplingFrequency -
        AcousticSamplingFrequency -
        )
        
    This functions serves to take all relevant motor characteristics and puts them in a 
    dictionary.
    This dictionary will serve as the building blocks for the rest of the functions.
    """
    
    #Arrange
    x = {
        'Name_ID': Name_ID,
        'Application': Application,
        'ModelNumber': ModelNumber,
        'SavingAlias': SavingAlias,
        'AccelerometerDataFilename': AccelerometerDataFilename,
        'AcousticDataFilename': AcousticDataFilename,
        'MLDataFilename': MLDataFilename,
        'Horsepower': Horsepower,
        'RatedVoltage': RatedVoltage,
        'ACorDC' : ACorDC,
        'NumberOfPolePairs': NumberOfPolePairs,
        'NumberofShafts': NumberofShafts,
        'ShaftSpeed': ShaftSpeed, #Shaft rotational speed [Hz], n
        'NumberOfRollingElements': NumberOfRollingElements, #No. of rolling elements [-], N
        'DiameterOfRollingElements': DiameterOfRollingElements, #Diameter of a rolling element [mm], Bd
        'PitchDiameter': PitchDiameter, #Pitch diameter [mm], Pd
        'ContactAngle': ContactAngle, #Contact angle [rad], Phi
        'AccelerometerSamplingFrequency': AccelerometerSamplingFrequency,
        'AcousticSamplingFrequency': AcousticSamplingFrequency
    }
    
    return x


# In[3]:


def System2CondensedForm(A2DResolution = None,VoltageMax = None,VoltageMin = None,                          SensorRunTime = None, AccelerationMax = None,AccelerationMin = None):
    
    """
    Returns a dictionary of all needed system, system, and 
    microcontroller information
    
    Inputs2CondensedForm(
        A2DResolution - 
        VoltageMax - 
        VoltageMin - 
        SensorRunTime - 
        AccelerationMax - 
        AccelerationMin - 
        )
        
    This functions serves to take all relevant motor characteristics 
    and puts them in a dictionary.
    This dictionary will serve as the building blocks for the rest 
    of the functions.
    """
    #Basic Calcualtations for System Constants
    VoltageRange = VoltageMax - VoltageMin
    VoltageQuantizedStep = VoltageRange / 2**A2DResolution
    AccelerationRange = AccelerationMax - AccelerationMin
    AccelerationQuantizedStep = AccelerationRange / VoltageRange
    
    #Arrange
    x = {
        'A2DResolution': A2DResolution,
        'SensorRunTime': SensorRunTime,
        'VoltageMax': VoltageMax,
        'VoltageMin': VoltageMin,
        'AccelerationMax': AccelerationMax,
        'AccelerationMin': AccelerationMin,
        'VoltageRange': VoltageRange,
        'VoltageQuantizedStep': VoltageQuantizedStep,
        'AccelerationRange': AccelerationRange,
        'AccelerationQuantizedStep': AccelerationQuantizedStep
    }
    
    return x


# In[4]:


def AllData2WorkingForm(Inputs2CondensedForm,System2CondensedForm,time, amp = None,                         voltage = None, acceleration = None,                        Channel1Time = None,Channel1Value = None):
    """
    Returns a dictionary of all needed information
    
    AllData2WorkingForm(
        Inputs2CondensedForm - 
        System2CondensedForm - 
        time - 
        amp - 
        voltage - 
        acceleration - 
        Channel1Time - 
        Channel1Value - 
    )
        
    This functions serves to take all relevant motor characteristics and puts them in a 
    dictionary.
    This dictionary will serve as the building blocks for the rest of the functions.
    """
    temp = copy.deepcopy(Inputs2CondensedForm)
    temp1 = copy.deepcopy(System2CondensedForm)
    
    #Get Extra Info
    try:
        NumberOfAccelerometerSamples = len(time)
        dt = 1/temp['AccelerometerSamplingFrequency']
        AccelerometerTmax = dt*NumberOfAccelerometerSamples
    except:
        NumberOfAccelerometerSamples = "N/A"
        AccelerometerTmax = "N/A"
    
    #Get Extra Info
    try:
        NumberOfAcousticSamples = len(Channel1Time)
        dt1 = 1/temp['AcousticSamplingFrequency']
        AcousticTmax = dt1*NumberOfAcousticSamples
    except:
        NumberOfAcousticSamples = "N/A"
        AcousticTmax = "N/A"
    
    #Arrange
    x = {
        'HomeDirectory': 'NotInstalled',
        'WorkingDirectory': 'NotInstalled',
        'NumberOfAccelerometerSamples': NumberOfAccelerometerSamples,
        'NumberOfAcousticSamples': NumberOfAcousticSamples,
        'AccelerometerSamplingTime': AccelerometerTmax,
        'AcousticSamplingTime': AcousticTmax,
        'AccelerometerTimeSeriesData': time,
        'AccelerometerBitAmplitudeData': amp,
        'AccelerometerVoltageData': voltage,
        'AccelerometerData': acceleration,
        'Channel1Time': Channel1Time,
        'Channel1Value': Channel1Value
    }
    temp.update(temp1)
    temp.update(x)
    
    return temp

