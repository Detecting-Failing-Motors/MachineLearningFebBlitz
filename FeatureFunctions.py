
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.signal import welch
from detect_peaks import detect_peaks
import matplotlib.pyplot as plt


# In[2]:


def BearingInformation(UserInput):
    """
    Returns a dictionary with Bearing Characteristic Frequencies
    
    BearingInfomation(
        UserInput - Dictionary of all info (see AllData2WorkingForm)
        )
        
    This functions calculates the bearing characteristic frequencies
    """
    #Get Needed Info
    n = UserInput['ShaftSpeed'] #Shaft rotational speed [Hz], n
    N = UserInput['NumberOfRollingElements']# No. of rolling elements [-], N
    Bd =  UserInput['DiameterOfRollingElements'] #Diameter of a rolling element [mm], Bd
    Pd = UserInput['PitchDiameter'] #Pitch diameter [mm], Pd
    phi = UserInput['ContactAngle'] #Contact angle [rad], Phi
                  
    #Calculate Bearing Frequncies using known equations
    try:
        xx   = Bd/Pd*np.cos(phi)
        BPFI = (N/2)*(1 + xx)*n
        BPFO = (N/2)*(1 - xx)*n
        BSF  = (Pd/(2*Bd))*(1-(xx)**2)*n
        FTF  = (1/2)*(1 - xx)*n
    except:
        BPFI = "N/A"
        BPFO = "N/A"
        BSF  = "N/A"
        FTF  = "N/A"
    
    #Arrange
    x = {
        "BPFI": BPFI,
        "BPFO": BPFO,
        "BSF":  BSF,
        "FTF":  FTF
    }
    return x


# In[3]:


def MotorInformation(UserInput):
    """
    Returns a Dictionary containg motor characteristics from UserInput
    that are needed for feature training
    
    MotorInformation(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
        
    Only valid for IMS dataset
    """
    if UserInput['ACorDC'] == "AC":
        temp = 0
    elif UserInput['ACorDC'] == "DC":
        temp = 1
    else:
        temp = "-1"
    
    x = {
        'Horsepower': UserInput['Horsepower'],
        'RatedVoltage': UserInput['RatedVoltage'],
        "ACorDC": temp,
        "NumberOfPoles": UserInput['NumberOfPolePairs'],
        "ShaftSpeed": UserInput['ShaftSpeed']
    }
    return x


# In[4]:


def TimeDomainInformation(UserInput):
    """
    Returns a dictionary with Time Domain Characteristics
    
    TimeDomainInformation(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
        
    This functions calculates the Time Domain Characteristics
    """
    #Get Needed Info
    sig = UserInput['AccelerometerData'] #Acceleration in terms of g's
    sig1 = UserInput['Channel1Value'] #Acoustic amplitude
    #Arrange
    x = {
        "AccelerometerRMS": np.mean(sig**2),
        "AccelerometerSTD": np.std(sig),
        "AccelerometerMean": np.mean(sig),
        "AccelerometerMax": np.max(sig),
        "AccelerometerMin": np.min(sig),
        "AccelerometerPeak-to-Peak": (np.max(sig) - np.min(sig)),
        "AccelerometerMax ABS": np.max(abs(sig)),
        "AccelerometerKurtosis": kurtosis(sig),
        "AccelerometerSkew": skew(sig),
        "AcousticRMS": np.mean(sig1**2),
        "AcousticSTD": np.std(sig1),
        "AcousticMean": np.mean(sig1),
        "AcousticMax": np.max(sig1),
        "AcousticMin": np.min(sig1),
        "AcousticPeak-to-Peak": (np.max(sig1) - np.min(sig1)),
        "AcousticMax ABS": np.max(abs(sig1)),
        "AcousticKurtosis": kurtosis(sig1),
        "AcousticSkew": skew(sig1),
    }

    return x


# In[5]:


def RemoveAllDCOffset(UserInput):
    """
    Returns a modified dictionary
    
    RemoveDCOffset(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
        
    This functions removes the dc bias from all value signal in the UserInput dictionary
    """
    #Copy
    temp = UserInput.copy()
    
    #Modify
    temp['AccelerometerBitAmplitudeData'] = temp["AccelerometerBitAmplitudeData"]     - np.mean(temp["AccelerometerBitAmplitudeData"])
    temp['AccelerometerVoltageData'] = temp["AccelerometerVoltageData"]     - np.mean(temp["AccelerometerVoltageData"])
    temp['AccelerometerData'] = temp["AccelerometerData"] - np.mean(temp["AccelerometerData"])
    temp['Channel1Value'] = temp["Channel1Value"] - np.mean(temp["Channel1Value"])
    
    return temp


# In[6]:


def Magnitude(Y):
    """
    Returns a float that is the magnitude of the array Y
    
    Magnitude(
     Y - an array of numbers to get the magnitude
    )
    """
    
    #Square
    mag = 0
    for i in range(0,len(Y)):
        mag = mag + Y[i]**2
        
    #Square Root
    mag = mag ** 0.5
    
    return mag


# In[7]:


def NormalizeAll(UserInput):
    """
    Returns a dictionary
    
    Normalize(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
    )
    """
    
    #Copy
    temp = UserInput.copy()
    
    #Normalize and Modify
    temp['AccelerometerBitAmplitudeData'] = temp["AccelerometerBitAmplitudeData"] / Magnitude(temp["AccelerometerBitAmplitudeData"])
    temp['AccelerometerVoltageData'] = temp["AccelerometerVoltageData"] / Magnitude(temp["AccelerometerVoltageData"])
    temp['AccelerometerData'] = temp["AccelerometerData"] / Magnitude(temp["AccelerometerData"])
    temp['Channel1Value'] = temp["Channel1Value"] / Magnitude(temp["Channel1Value"])


    return temp


# In[8]:


def getTESTDataFrame(UserInput):
    """
    Returns a Dataframe that does not need the state
    
    getTESTDataFrame(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
    
    This function generates a dataframe without knowing its state
    This function calls functions in the consistent order
    """
    #Call specific function order for consistency 
    UserInput1 = UserInput.copy()
    UserInput2 = RemoveAllDCOffset(UserInput1)
    UserInput3 = NormalizeAll(UserInput2)
    BearingInfo = BearingInformation(UserInput3)
    TimeDomainInfo = TimeDomainInformation(UserInput3)
    FrequecyDomainInfo = FrequencyDomainInformation(UserInput3)
    MotorInfo = MotorInformation(UserInput3)
    
    #Arrange (with no state info)
    Features = {**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    
    return Features 


# In[9]:


def FourierTransform(UserInput):
    """
    Returns a dictionary what contains the frequency and frequency amplitude arrays
    
    FourierTransform(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
    )
    
    The functions perfroms fast fourier transform on the UserInput Signal 
    Data of Interest
    """
    
    #Get Needed Info for Accelerometer
    NumberOfSamples1 = UserInput['NumberOfAccelerometerSamples']
    Tmax1 = UserInput['AccelerometerSamplingTime']
    sig1 = UserInput['AccelerometerData']
    
    #Fourier Transform for Accelerometer
    frq1 = np.arange(NumberOfSamples1)/(Tmax1)# two sides frequency range
    frq1 = frq1[range(int(NumberOfSamples1/(2)))] # one side frequency range
    Y1 = abs(np.fft.fft(sig1))/NumberOfSamples1 # fft computing and normalization
    Y1 = Y1[range(int(NumberOfSamples1/2))]
    
    #Get Needed Info for Acoustic
    NumberOfSamples2 = UserInput['NumberOfAcousticSamples']
    Tmax2 = UserInput['AcousticSamplingTime']
    sig2 = UserInput['Channel1Value']
    
    #Fourier Transform for Acoustic
    frq2 = np.arange(NumberOfSamples2)/(Tmax2)# two sides frequency range
    frq2 = frq2[range(int(NumberOfSamples2/(2)))] # one side frequency range
    Y2 = abs(np.fft.fft(sig2))/NumberOfSamples2 # fft computing and normalization
    Y2 = Y2[range(int(NumberOfSamples2/2))]
    
    #Arrange
    x = {
        "AccelerometerFrequency":frq1,
        "Accelerometer Freq. Amp.": Y1,
        "Acoustic Frequency":frq2,
        "Acoustic Freq. Amp.": Y2
        }
    
    return x


# In[10]:


def GetPSDValues(UserInput):
    """
    Returns a dictionary that contains the frequency and the frequency amplitude arrays
    
    get_psd_values(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
    )
    
    The functions perfroms power spectrum density on the UserInput Signal 
    Data of Interest
    """
    
    #Get Needed Info for Accelerometer
    sig1 = UserInput['AccelerometerData']
    SamplingFrequency1 = UserInput['AccelerometerSamplingFrequency']
    
    #Perform PSD for Accelerometer
    frq1, PSD1 = welch(sig1, fs=SamplingFrequency1)
    
    #Get Needed Info for Acoustic
    sig2 = UserInput['Channel1Value']
    SamplingFrequency2 = UserInput['AcousticSamplingFrequency']
    
    #Perform PSD for Acoustic
    frq2, PSD2 = welch(sig2, fs=SamplingFrequency2)
    
    #Arrange
    x = {
        "AccelerometerPSDFrequency":frq1,
        "AccelerometerPSD": PSD1,
        "AcousticPSDFrequency":frq2,
        "AcousticPSD": PSD2
        }
    
    return x


# In[11]:


def autocorr(x):
    """
    Taken from:
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    
    Returns the autocorrelation of the signal x
    
    autocorr(
        x - signal of interest
        )
    
    This functions performs correlation
    """
    result = np.correlate(x, x, mode='full')
    return result[len(result)//2:]


# In[12]:


def GetAutocorrValues(UserInput):
    """
    Modified from: 
    https://ipython-books.github.io/103-computing-the-autocorrelation-of-a-time-series/
    
    Returns a dictionary
    
    get_autocorr_values(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
    )
    
    The functions perfroms autocorrelation on the UserInput Signal 
    Data of Interest
    """
    #Get needed info for Accelerometer
    NumberOfSamples1 = UserInput['NumberOfAccelerometerSamples']
    Tmax1 = UserInput['AccelerometerSamplingTime']
    sig1 = UserInput['AccelerometerData']
    
    #Call correlation function
    AutocorrValues1 = autocorr(sig1)
    #Arrange XValues
    XValues1 = np.array([Tmax1 * jj for jj in range(0, NumberOfSamples1)])
    
    #Get Needed Info for Acoustic
    NumberOfSamples2 = UserInput['NumberOfAcousticSamples']
    Tmax2 = UserInput['AcousticSamplingTime']
    sig2 = UserInput['Channel1Value']
    
    #Call correlation function
    AutocorrValues2 = autocorr(sig2)
    #Arrange XValues
    XValues2 = np.array([Tmax2 * jj for jj in range(0, NumberOfSamples2)])

    #Arrange
    x = {
        "AccelerometerXValues": XValues1,
        "AccelerometerAutocorrValues": AutocorrValues1,
        "AcousticXValues": XValues2,
        "AcousticAutocorrValues": AutocorrValues2
        }
    
    return x


# In[13]:


def GetSortedPeak(X,Y):
    """
    SubFunction for FrequencyDomainInformation
    
    Returns Amplitude of Y, Loctation
    
    GetSortedPeak(
        X - Independent Variable
        Y - Dependent Variable
        )
        
    Uses detect_peaks function taken from Github:
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    
    Get the indices of relevant peaks
    Then Returns the amplitude,location of the relevant peaks
    """
    #Original
    """
    #Set Parameters
    min_peak_height = 0.1 * np.nanmax(Y) #Original
    threshold = 0.05 * np.nanmax(Y) #Original
    
    #Get indices of peak
    peak = detect_peaks(Y,edge = 'rising',mph = min_peak_height, mpd = 2, threshold = threshold ) #Original
    """
    #NEW
    #Set Parameters
    Ymag = Magnitude(Y)
    Ynew = Y/Ymag
    min_peak_height = .04
    threshold = 0.15*np.std(Ynew)
    
    #Get indices of peak
    peak = detect_peaks(Ynew,edge = 'rising',mph = min_peak_height, mpd = 5, threshold = threshold )
    
    #Get values corresponding to indices 
    m = []
    mm = []
    for i in peak:
        m.append(Y[i]) 
        mm.append(X[i])

    #Sort arcording to the amplitude
    mmm = np.argsort(m)
    n = []
    nn = []
    for i in mmm:
        n.append(m[i])
        nn.append(mm[i])
    
    #Sort in Descending Amplitdue while keeping locations matched
    n  = n[::-1] #amplitude
    nn = nn[::-1] #location
    
    #Arrange
    return n, nn


# In[14]:


def FrequencyDomainInformation(UserInput):
    """
    Returns a dictionary with Frequency Domain Characteristics
    Top 5 frequncy and amplitudes for:
    fft
    psd
    correlation
    
    FrequencyDomainInformation(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
        
    Note: 77777 refers to a blank cell
        We had to fill in blank cells to work with pandas dataframe
    """
    
    #*********************************
    #AS OF 2/9/2020 Autocorr Functionality was removed
    #*********************************
    
    #Call FFT, PSD, and Correlation Values
    x1 = FourierTransform(UserInput)
    x2 = GetPSDValues(UserInput)
    x3 = GetAutocorrValues(UserInput)
    
    #Accelerometer
    FTamp1,FTfreq1 = GetSortedPeak(x1["AccelerometerFrequency"],x1["Accelerometer Freq. Amp."])
    PSDamp1,PSDfreq1 = GetSortedPeak(x2["AccelerometerPSDFrequency"],x2["AccelerometerPSD"])
    #Cor1,CorTime1 = GetSortedPeak(x3["AccelerometerXValues"],x3["AccelerometerAutocorrValues"])
    
    #Take Care of Empty Values
    while len(FTamp1) <= 5:
        FTamp1.append(['77777'])
    while len(FTfreq1) <= 5:
        FTfreq1.append(['77777'])
    while len(PSDamp1) <= 5:
        PSDamp1.append(['77777'])
    while len(PSDfreq1) <= 5:
        PSDfreq1.append(['77777'])
    #while len(Cor1) <= 5:
    #    Cor1.append(['77777'])
    #while len(CorTime1) <= 5:
    #    CorTime1.append(['77777'])
    
    #Acoustic
    FTamp2,FTfreq2 = GetSortedPeak(x1["Acoustic Frequency"],x1["Acoustic Freq. Amp."])
    PSDamp2,PSDfreq2 = GetSortedPeak(x2["AcousticPSDFrequency"],x2["AcousticPSD"])
    #Cor2,CorTime2 = GetSortedPeak(x3["AcousticXValues"],x3["AcousticAutocorrValues"])

    #Take Care of Empty Values
    while len(FTamp2) <= 5:
        FTamp2.append(['77777'])
    while len(FTfreq2) <= 5:
        FTfreq2.append(['77777'])
    while len(PSDamp2) <= 5:
        PSDamp2.append(['77777'])
    while len(PSDfreq2) <= 5:
        PSDfreq2.append(['77777'])
    #while len(Cor2) <= 5:
    #    Cor2.append(['77777'])
    #while len(CorTime2) <= 5:
    #    CorTime2.append(['77777'])
    
    #Arrange
    accelerometer = {
        "Accelerometer FFT Frq @ Peak 1": FTfreq1[0],
        "Accelerometer FFT Frq @ Peak 2": FTfreq1[1],
        "Accelerometer FFT Frq @ Peak 3": FTfreq1[2],
        "Accelerometer FFT Frq @ Peak 4": FTfreq1[3],
        "Accelerometer FFT Frq @ Peak 5": FTfreq1[4],
        "Accelerometer FFT Amp @ Peak 1": FTamp1[0],
        "Accelerometer FFT Amp @ Peak 2": FTamp1[1],
        "Accelerometer FFT Amp @ Peak 3": FTamp1[2],
        "Accelerometer FFT Amp @ Peak 4": FTamp1[3],
        "Accelerometer FFT Amp @ Peak 5": FTamp1[4],
        "Accelerometer PSD Frq @ Peak 1": PSDfreq1[0],
        "Accelerometer PSD Frq @ Peak 2": PSDfreq1[1],
        "Accelerometer PSD Frq @ Peak 3": PSDfreq1[2],
        "Accelerometer PSD Frq @ Peak 4": PSDfreq1[3],
        "Accelerometer PSD Frq @ Peak 5": PSDfreq1[4],
        "Accelerometer PSD Amp @ Peak 1": PSDamp1[0],
        "Accelerometer PSD Amp @ Peak 2": PSDamp1[1],
        "Accelerometer PSD Amp @ Peak 3": PSDamp1[2],
        "Accelerometer PSD Amp @ Peak 4": PSDamp1[3],
        "Accelerometer PSD Amp @ Peak 5": PSDamp1[4]
        }
    """
    "Accelerometer Autocorrelate Time @ Peak 1": CorTime1[0],
    "Accelerometer Autocorrelate Time @ Peak 2": CorTime1[1],
    "Accelerometer Autocorrelate Time @ Peak 3": CorTime1[2],
    "Accelerometer Autocorrelate Time @ Peak 4": CorTime1[3],
    "Accelerometer Autocorrelate Time @ Peak 5": CorTime1[4],
    "Accelerometer Autocorrelate @ Peak 1": Cor1[0],
    "Accelerometer Autocorrelate @ Peak 2": Cor1[1],
    "Accelerometer Autocorrelate @ Peak 3": Cor1[2],
    "Accelerometer Autocorrelate @ Peak 4": Cor1[3],
    "Accelerometer Autocorrelate @ Peak 5": Cor1[4]
    }
    """
    
    
    acoustic = {
        "Acoustic FFT Frq @ Peak 1": FTfreq2[0],
        "Acoustic FFT Frq @ Peak 2": FTfreq2[1],
        "Acoustic FFT Frq @ Peak 3": FTfreq2[2],
        "Acoustic FFT Frq @ Peak 4": FTfreq2[3],
        "Acoustic FFT Frq @ Peak 5": FTfreq2[4],
        "Acoustic FFT Amp @ Peak 1": FTamp2[0],
        "Acoustic FFT Amp @ Peak 2": FTamp2[1],
        "Acoustic FFT Amp @ Peak 3": FTamp2[2],
        "Acoustic FFT Amp @ Peak 4": FTamp2[3],
        "Acoustic FFT Amp @ Peak 5": FTamp2[4],
        "Acoustic PSD Frq @ Peak 1": PSDfreq2[0],
        "Acoustic PSD Frq @ Peak 2": PSDfreq2[1],
        "Acoustic PSD Frq @ Peak 3": PSDfreq2[2],
        "Acoustic PSD Frq @ Peak 4": PSDfreq2[3],
        "Acoustic PSD Frq @ Peak 5": PSDfreq2[4],
        "Acoustic PSD Amp @ Peak 1": PSDamp2[0],
        "Acoustic PSD Amp @ Peak 2": PSDamp2[1],
        "Acoustic PSD Amp @ Peak 3": PSDamp2[2],
        "Acoustic PSD Amp @ Peak 4": PSDamp2[3],
        "Acoustic PSD Amp @ Peak 5": PSDamp2[4]
        }
    """
    "Acoustic Autocorrelate Time @ Peak 1": CorTime2[0],
    "Acoustic Autocorrelate Time @ Peak 2": CorTime2[1],
    "Acoustic Autocorrelate Time @ Peak 3": CorTime2[2],
    "Acoustic Autocorrelate Time @ Peak 4": CorTime2[3],
    "Acoustic Autocorrelate Time @ Peak 5": CorTime2[4],
    "Acoustic Autocorrelate @ Peak 1": Cor2[0],
    "Acoustic Autocorrelate @ Peak 2": Cor2[1],
    "Acoustic Autocorrelate @ Peak 3": Cor2[2],
    "Acoustic Autocorrelate @ Peak 4": Cor2[3],
    "Acoustic Autocorrelate @ Peak 5": Cor2[4]
    }
    """
    
    
    return {**accelerometer,**acoustic}


# In[15]:


def getQuickPlot(X,Y,xlabel=None,ylabel=None,Title=None):
    """
    Subfunction of getGraphs
    Returns a figure
    
    getQuickPlot(
        X - Data for independent variable
        Y - Data for dependent variable
        xlabel - X-axis label
        ylabel - Y-axis label
        Title - Title of figure
        )
    
    Performs plt.plot
    """
    
    #Plot
    fig = plt.figure()
    plt.plot(X,Y,c = np.random.rand(3,))
    if xlabel != None:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel != None:
        plt.ylabel(ylabel, fontsize=12)
    if Title != None:
        plt.title(Title)
    plt.grid(True)
    
    return fig


# In[16]:


def getGraphs(UserInput):
    """
    Returns figure information for 12 graphs
    
    getGraphs(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
    
    """
    #REMOVED AUTOCORR FUNCTIONALITY 2/11/2020
    
    #Perform FFT, PSD, Correlation, DC Offset
    UserInput1 = RemoveAllDCOffset(UserInput)
    UserInput2 = NormalizeAll(UserInput1)
    x1 = FourierTransform(UserInput2)
    x2 = GetPSDValues(UserInput2)
    #x3 = GetAutocorrValues(UserInput2)
    
    #ACCELEROMETER PLOTTING INFO
    plotinfo = getPlotInfo(UserInput['AccelerometerTimeSeriesData'],UserInput['AccelerometerData'],                          "time (s)","Amplitude [g's]","Accelerometer Raw Data")
    plot2info = getPlotInfo(UserInput1['AccelerometerTimeSeriesData'],UserInput1['AccelerometerData'],                          "time (s)","Amplitude [g's]","Accelerometer Raw Data w/ Removed DC Offset")
    plot3info = getPlotInfo(UserInput2['AccelerometerTimeSeriesData'],UserInput2['AccelerometerData'],                          "time (s)","Amplitude [g's/g's]","Accelerometer Normalized Raw Data")
    plot4info = getPlotInfo(x1["AccelerometerFrequency"],x1["Accelerometer Freq. Amp."],                           'Frequency [Hz]',"FFT Amp [g\'s]","Accelerometer FFT")
    plot5info = getPlotInfo(x2["AccelerometerPSDFrequency"],x2["AccelerometerPSD"],'Frequency [Hz]',                           'PSD [g\'s**2 / Hz]',"Accelerometer PSD")
    #plot6info = getPlotInfo(x3["AccelerometerXValues"],x3["AccelerometerAutocorrValues"],'time delay [s]',\
    #                       "Autocorrelation amplitude","Accelerometer Autocorrelation")
    
    #ACOUSTIC PLOTTING INFO
    plot7info = getPlotInfo(UserInput['Channel1Time'],UserInput['Channel1Value'],                          "time (s)","Amplitude","Acoustic Raw Data")
    plot8info = getPlotInfo(UserInput1['Channel1Time'],UserInput1['Channel1Value'],                          "time (s)","Amplitude","Acoustic Raw Data w/ Removed DC Offset")
    plot9info = getPlotInfo(UserInput2['Channel1Time'],UserInput2['Channel1Value'],                          "time (s)","Amplitude [g's/g's]","Acoustic Normalized Raw Data")
    plot10info = getPlotInfo(x1["Acoustic Frequency"],x1["Acoustic Freq. Amp."],                           'Frequency [Hz]',"FFT Amp [g\'s]","Acoustic FFT")
    plot11info = getPlotInfo(x2["AcousticPSDFrequency"],x2["AcousticPSD"],'Frequency [Hz]',                           'PSD [g\'s**2 / Hz]',"Acoustic PSD")
    #plot12info = getPlotInfo(x3["AcousticXValues"],x3["AcousticAutocorrValues"],'time delay [s]',\
    #                       "Autocorrelation amplitude","Acoustic Autocorrelation")

    
    return plotinfo,plot2info,plot3info,plot4info,plot5info,            plot7info,plot8info,plot9info,plot10info,plot11info


# In[17]:


def getPlotInfo(x,y,xlabel,ylabel,title):
    """
    Returns an array of all needed info for quick plot
    
    getPlotInfo(
        X - Data for independent variable
        Y - Data for dependent variable
        xlabel - X-axis label
        ylabel - Y-axis label
        Title - Title of figure
        )
        
    """
    PlotInfo = []
    PlotInfo.append(x)
    PlotInfo.append(y)
    PlotInfo.append(xlabel)
    PlotInfo.append(ylabel)
    PlotInfo.append(title)
    
    return PlotInfo


# In[18]:


def truncate(f, n):
    '''https://stackoverflow.com/questions/783897/truncating-floats-in-python/51172324#51172324'''
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

