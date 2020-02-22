
# coding: utf-8

# In[1]:


from random import random
import pandas as pd
import FeatureFunctions as FF
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# In[2]:


def getTESTMatrix(FeatureArray):
    """
    Returns an array that can be directly plugged into Predict()
    
    getTestMatrix(
        FeatureArray - Output of getTESTDataFrame(UserInput)
    )
    """
    
    #Correct the type of data
    return FeatureArray.values[:,0:(FeatureArray.shape[1]-1)]


# In[3]:


def StateDict():
    """
    Returns a dictionary "Key": Value
    
    StateDict()
    
    The key is the more verbose describtion of the output state of the training data
    The value is the int that is defined to relate to the corresponding key. 
    """
    
    State2Int = {
        "Suspect": 0,
        "Normal": 1,
        "Failure": 2,
        "ERROR": 77777
    }
    
    return State2Int


# In[4]:


def RandomStateInformation(UserInput):
    """
    Returns a Dictionary of Motor State
    
    RandomStateInformation(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
    
    This function is used to generate a random outcome for the limited training data.
    This function is only intended to aid in generating the training data.
    """
    
    rando = random()
    
    if 0 <= rando and rando < 0.3:
        m = "Suspect"
    elif 0.3 <= rando and rando < 0.6:
        m = "Normal"
    elif 0.6 <= rando and rando <= 1.0:
        m = "Failure"
    else :
        m = "ERROR"
    
    State2Int = StateDict()
    
    #Arrange
    x = {
        "State": State2Int[m]
    }
    return x


# In[5]:


def getCompleteDataFrame(UserInput):
    """
    Returns a Dataframe for sample
    
    getCompleteDataFrame(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        )
    
    This function is used to generate a known outcome for the training data.
    This function is only intended to aid in generating the training data.
    """
    #*************ONLY  USED WITH GENERATING TRAINGING FILE **********************
    
    #import FeatureFunctions as FF
    
    #Call specific function order for consistency 
    UserInput1 = UserInput.copy()
    UserInput2 = FF.RemoveAllDCOffset(UserInput1)
    UserInput3 = FF.NormalizeAll(UserInput2)
    BearingInfo = FF.BearingInformation(UserInput3)
    TimeDomainInfo = FF.TimeDomainInformation(UserInput3)
    FrequecyDomainInfo = FF.FrequencyDomainInformation(UserInput3)
    StateInfo = RandomStateInformation(UserInput3)
    MotorInfo = FF.MotorInformation(UserInput3)
    
    #Arrange
    Features = {**StateInfo,**MotorInfo,**BearingInfo,**TimeDomainInfo,**FrequecyDomainInfo}
    Features = pd.DataFrame(Features, index=[0])
    return Features 


# In[6]:


def GenerateTrainingFile(UserInput,string):
    """
    returns True upon completition
    
    GenerateTrainingFile(
        UserInput - Dictionary of relevant info (see AllData2WorkingForm)
        string - filename of CSV files (include CSV extension)
    )
    
    """


    m = [] 
    #Number of Rows
    NumberofRows = 75
    for i in range(0,NumberofRows):

        Features = getCompleteDataFrame(UserInput)
        m.append(Features.values[0,:])

    ColumnTitle = Features.columns
    DF = pd.DataFrame(m,columns = ColumnTitle)
    DF.to_csv(string)


# In[7]:


def TrainModel(X_train,Y_train):
    """
    Returns a classifier that has been fit
    
    TrainModel(
        X_train - Training Data
        Y_train - Results of Training Data for supervised learning
        )
    
    Currently only fits RandomForestClassifier (winning classifier)
    """
    
    #Fit final model
    classifier = RandomForestClassifier(min_samples_split= 10 ,n_estimators = 200)
    classifier = classifier.fit(X_train, Y_train)
    
    
    return classifier


# In[9]:


def get_key(value,dictionary): 
    """
    
    Modified from:
    https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
    """
    
    #Return key if the value in dictionary is the predetermined value
    result = []
    for x in value:
        for key, val in dictionary.items(): 
             if val == x:
                result.append( key )

    return result


# In[8]:


def PredictModel(classifier,X_test):
    """
    Returns a tuple of prediction in integer form, string form
    
    PredictModel(
        classifier - fitted classifier
        X_test - data to be tested
        )
    """
    #Get key-value relationship
    State2Int = StateDict()
    
    #Predict
    Y_test_pred = classifier.predict(X_test)
    
    #Get string (key) of the prediction
    Y_test_pred_string = get_key(Y_test_pred,State2Int)
    
    return Y_test_pred, Y_test_pred_string


# In[10]:


def PredictProbModel(classifier,X_test):
    """
    Returns a prediction probability (out of 100 not 1)
    
    PredictModel(
        classifier - fitted classifier
        X_test - data to be tested
        )
    """
    """
    COMBINE WITH PREDICT MODEL GOING FORWARD
    """
    
    #Get the probability prediction
    Y_test_pred_proba = classifier.predict_proba(X_test)
    
    return Y_test_pred_proba*100


# In[11]:


def GetTrainingData(UserInput):
    """
    Returns X_train, Y_train
    
    getGraphs(UserInput)
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        )
    
    This returns the training and test sets
    """
    
    #Find training file name and read it
    dataset = pd.read_csv(UserInput['MLDataFilename'],header = 0,index_col = 0)

    #Return the entire sets
    X_train = dataset.values[:,1:(dataset.shape[1]-1)]
    Y_train = dataset.values[:,0]
    
    return X_train, Y_train, dataset


# In[12]:


def GetSplitTrainingData(UserInput, seed = 6):
    """
    Returns an X_train, X_test, Y_train, Y_test
    
    getGraphs(UserInput)
        UserInput - Dictionary of relevant info (see UserInputs2WorkingForm)
        seed - random number for splitting of test and trainig (default = 6)
        )
    
    This returns the training and test sets
    """
    
    #Find training file name and read it
    dataset = pd.read_csv(UserInput['MLDataFilename'],header = 0,index_col = 0)


    #Get the values
    X = dataset.values[:,1:(dataset.shape[1]-1)]
    Y = dataset.values[:,0]
    
    #Set splitting parameters
    validation_size = 0.20
    seed = seed
    
    #Split data
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) 
    
    return X_train, X_test, Y_train, Y_test


# In[13]:


def GetReducedFeaturesFromDataFrame(df):
    """
    Returns a dataframe with important features
    
    GetReducedFeaturesFromDataFrame(
        df - dataframe of the getTestDataFrame that contains the data to be analyzed
        )
    """
    #Extract Features
    x = df[['AccelerometerRMS','FTF','AccelerometerMax ABS',            'AcousticRMS','AcousticMax ABS']]
    
    return x

