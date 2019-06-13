import numpy as np
import numpy.ma as ma
import math as mt
import datetime
import sklearn.metrics
import statsmodels.stats.inter_rater
import pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn import tree,preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 30)

import time
start = time.time()

def kappa(labels, predictions):
    sh=np.shape(predictions)
    if len(sh)>1:
        pred_round   = np.round(np.mean(predictions[:,:],axis=1))  # accepts multiple predictions per image (requires a 2D input)
    elif len(sh)==1:
        pred_round   = np.round(predictions)
    F1_score     = sklearn.metrics.f1_score(labels, pred_round)  
    conf_matrix  = sklearn.metrics.confusion_matrix(labels,pred_round)
    kappa        = statsmodels.stats.inter_rater.cohens_kappa(conf_matrix, weights=None, return_results=True, wt=None)
    AUC=roc_auc_score(labels,pred_round)
    fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  #false positives
    fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)   #false negatives
    tp = np.diag(conf_matrix)  #true positives
    tn = np.sum(conf_matrix,axis=(0,1)) - (fp + fn + tp)   #true negatives
    
    sensi=tp/(tp+fn)
    speci=tn/(tn+fp)
    PPV=tp/(tp+fp)  #positive predictive value
    NPV=tn/(tn+fn)  #negative predictive value
    return(F1_score, kappa,AUC,PPV[1],NPV[1],sensi[1],speci[1])
    
def con_lab(dframe):
    '''convert the relevant labels to binary'''
    column=dframe['SubjectDiabetesStatus']  #select the column with labels
    labels=[0,3]   #labels displaying healthy and T2D
    s1=(column==labels[0]) #indices of healthy patients
    s2=(column==labels[1])  #indices of T2D
    mask = [any(tup) for tup in zip(s1, s2)]  #add the relavant indices together
    y_true=column.loc[mask]  #mask the column to extract patients
    lut = dict(zip(labels, [0,1]))  #create dictionary
    y_true=y_true.map(lut)  #convert labels to binary
    return y_true,mask
    
def averaging(feats,y_true,num):
    '''average the probabilities subject-level and evaluate'''
    pred=feats[feats.columns[0:num]]   #only take the means in the DataFrame
    (rows,col)=pred.shape   #get shape
    means=np.zeros((rows))   #reserve memory for the averages
    for i in range(0,rows):   #go over the patients
        p=pred.iloc[i]   #get the patient row
        means[i]=np.mean(p.loc[p!=0])   #only take non-zero values and calculate the average
    (F1,kap,AUC,PPV,NPV,sensi,speci)=kappa(y_true,means)  #evaluate when averaging the predictions
    return F1,kap['kappa'],AUC,PPV,NPV,sensi,speci

def first_only(feats,y_true,num):
    '''Only take the first prediction for each subject and evaluate scores'''
    (F1,kap,AUC,PPV,NPV,sensi,speci)=kappa(y_true,feats[feats.columns[0]])  #evaluate when only taking the first mean
    return F1,kap['kappa'],AUC,PPV,NPV,sensi,speci

def major(feats,y_true,num):
    '''Majority Voting to determine predicted label and evaluate'''
    pred=feats[feats.columns[0:num]]  #only take the means in the DataFrame
    (rows,col)=pred.shape  #get shape
    preds=np.zeros((rows))   #reserve memory for predicted labels
    for i in range(0,rows):  #go over the patients
        p=pred.iloc[i]   #take the patient
        p=p.loc[p!=0]   #only take non-zero means
        rnd=np.round(p)   #round the means to binary
        sm=np.sum(rnd)   #sum binary values
        bol=(sm>=(np.size(rnd)/2))   #If number of positives is in excess or voting is equal than give a True value
        preds[i]=bol.astype(int)  #boolean to an integer, so binary
    (F1,kap,AUC,PPV,NPV,sensi,speci)=kappa(y_true,preds)  #calculate scores
    return F1,kap['kappa'],AUC,PPV,NPV,sensi,speci
    
def display_results(methods,**kwargs):
    '''display the scores for all methods'''
    if kwargs is not None:  #if any keyword arguments are give continue
        frame=pd.DataFrame(methods,columns=['Methods']) #set up a dataframe
        for metric, value in kwargs.items():   #unpack the dictionary of keyword arguments
            frame[metric.replace('_',' ')]=value #add new column to the frame
        #print(frame)            
    return frame

def prepare_data(dframe,predictions,padding,normalize,method,certainty):
    '''preprocessing of the data to serve as features for the classifiers'''
    y,mask=con_lab(dframe)   #get the labels and indexes of correct patients
    predictions=predictions[mask,:]   #only take corresponding predictions
    dframe=dframe.loc[mask,:]     #only take correct indexes
    idx=dframe['SubjectID3'].value_counts()    #all the patient ID's and number of occurences in the list
    #get number of means which will be in list and names of features
    if method=='Eyes':
        mx=2  #number of eyes
        names=['Mean left','Mean right','Std left','Std right']  #feature names
    elif method=='All': 
        mx=idx.max()  #maximum number of occurences
        names=['Prob 1','Prob 2','Prob 3','Prob 4','Prob 5','Prob 6','Prob 7','Prob 8','Prob 9','Prob 10','Prob 11','Prob 12','Std 1','Std 2','Std 3','Std 4','Std 5','Std 6','Std 7','Std 8','Std 9','Std 10','Std 11','Std 12']
        names=names[0:mx]+names[12:12+mx]
    elif method=='Center':
        mx=2   #OD or Fovea
        names=['Mean OD','Mean Fovea','Std OD','Std Fovea']
    elif method=='Both':
        mx=4  #number of OD/Fovea with left/right combinations
        names=['Mean OD/Left','Mean Fovea/Left','Mean OD/Right','Mean Fovea/Right','Std OD/Left','Std Fovea/Left','Std OD/Right','Std Fovea/Right']
    features=np.zeros((len(idx),2*mx))    #make space in memory for all means and std's of the predictions
    unq=dframe['SubjectID3'].unique()   #all unqiue ID's
    adt=0   #will be used to take the correct rows of the prediction matrix
    ages=np.zeros((len(idx),1))  #reserve memory for the ages
    y_true=np.zeros((len(idx),1))  #reserve memory for the true labels
    gender=np.zeros((len(idx),1))
    y_cert=np.zeros((len(idx),1))
    for index,pat in enumerate(unq):   #go over all patients
        part=dframe.loc[dframe['SubjectID3']==pat]    #only take the part of the DF of this patient
        ages[index]=int(part['SubjectAge'].iloc[0])   #include the age of this patient
        gender[index]=part['SubjectGender'].iloc[0]
        y_true[index]=int(y.loc[(dframe['SubjectID3']==pat)].iloc[0]) #set the correct label of this patient
        if certainty:
            preds=predictions[adt:adt+idx.loc[pat],:]   #all predictions of this patient
            top=np.sum(preds>0.8,axis=1)
            bot=np.sum(preds<0.2,axis=1)
            if np.sum(top>2)>0 and np.sum(bot>2)==0:
                y_cert[index]=1
            elif np.sum(bot>2)>0 and np.sum(top>2)==0:
                y_cert[index]=0
            else:
                y_cert[index]=2
        if method=='Eyes':  #mean and standard deviation of eyes seperately to serve as features
            preds=predictions[adt:adt+idx.loc[pat],:]   #all predictions of this patient
            il=(part['Eye']=='Left')   #indexes of left eye
            ir=(part['Eye']=='Right')   #indexes of right eye
            left=preds[il,:]   #predictions left eye
            right=preds[ir,:]   #predictions right eye
            means=[np.mean(np.mean(left,axis=1)),np.mean(np.mean(right,axis=1))]  #mean of left and right eye
            stands=[np.std(np.mean(left,axis=1)),np.std(np.mean(right,axis=1))]  #std of left and right eye
            if padding=='Average':
                means = [means[abs(i-1)] if mt.isnan(x) else x for i,x in enumerate(means)]   #replace Nan values with mean value other eye
                stands = [stands[abs(i-1)] if mt.isnan(x) else x for i,x in enumerate(stands)]  #replace Nan values with std value of other eye
            elif padding=='Zero':
                means = [0 if mt.isnan(x) else x for i,x in enumerate(means)]   #replace Nan values with 0
                stands = [0 if mt.isnan(x) else x for i,x in enumerate(stands)]  #replace Nan values with 0
            
        elif method=='Both':
            preds=predictions[adt:adt+idx.loc[pat],:]   #all predictions of this patient
            il=(part['Eye']=='Left')   #indexes of left eye
            ir=(part['Eye']=='Right')   #indexes of right eye
            iod=(part['ODorFoveaCentered']=='OD')   #indexes of OD center
            ifo=(part['ODorFoveaCentered']=='Fovea')   #indexes of Fovea center
            OD_L=(il & iod) #OD centering - left eye
            FOV_L=(il % ifo)   #Fovea Centering - Left eye
            OD_R=(ir % iod)   #OD centering - Right eye
            FOV_R=(ir & ifo)   #Fovea Centering - Right Eye
            means=[np.mean(np.mean(preds[OD_L,:],axis=1)),np.mean(np.mean(preds[FOV_L,:],axis=1)),np.mean(np.mean(preds[OD_R,:],axis=1)),np.mean(np.mean(preds[FOV_R,:],axis=1))]  #mean features
            stands=[np.std(np.mean(preds[OD_L,:],axis=1)),np.std(np.mean(preds[FOV_L,:],axis=1)),np.std(np.mean(preds[OD_R,:],axis=1)),np.std(np.mean(preds[FOV_R,:],axis=1))]   #std features
            if padding=='Zero':
                means=np.where(np.isnan(means),0,means)   #replace nan values with a 0
                stands=np.where(np.isnan(stands),0,stands)  #replace nan values with a 0
                
            elif padding=='Average':
                nans=np.isnan(means)   #places which contain Nan values
                means=np.where(np.isnan(means),np.mean(ma.masked_array(means,mask=nans)),means)   #replace Nan values with the mean of the means that are present
                stands=np.where(np.isnan(stands),np.mean(ma.masked_array(stands,mask=nans)),stands)  #replace Nan values with the mean of the std's that are present

        elif method=='Center':  #mean and standard deviation of centers seperately to serve as features
            preds=predictions[adt:adt+idx.loc[pat],:]   #all predictions of this patient
            iod=(part['ODorFoveaCentered']=='OD')   #indexes of OD center
            ifo=(part['ODorFoveaCentered']=='Fovea')   #indexes of Fovea center
            od=preds[iod,:]   #predictions OD center
            fov=preds[ifo,:]   #predictions Fovea center
            means=[np.mean(np.mean(od,axis=1)),np.mean(np.mean(fov,axis=1))]  #mean of OD and Fovea center
            stands=[np.std(np.mean(od,axis=1)),np.std(np.mean(fov,axis=1))]  #std of OD and Fovea center
            if padding=='Average':
                means = [means[abs(i-1)] if mt.isnan(x) else x for i,x in enumerate(means)]   #replace Nan values with mean value other eye
                stands = [stands[abs(i-1)] if mt.isnan(x) else x for i,x in enumerate(stands)]  #replace Nan values with std value of other eye
            elif padding=='Zero':
                means = [0 if mt.isnan(x) else x for i,x in enumerate(means)]   #replace Nan values with 0
                stands = [0 if mt.isnan(x) else x for i,x in enumerate(stands)]  #replace Nan values with 0
            
        elif method=='All':
            means=np.mean(predictions[adt:adt+idx.loc[pat],:],axis=1)  #take all of the mean predictions of this patient
            stands=np.std(predictions[adt:adt+idx.loc[pat],:],axis=1)  #take all of the std's of the patient predictions
        adt=adt+idx.loc[pat]  #update the variabel 
        features[index,0:len(means)]=np.reshape(means,(len(means)))   #set the means of this patient in the matrix
        features[index,mx:mx+len(stands)]=np.reshape(stands,(len(stands))) #set the standard deviations of this patient in the second half of this patient
        if padding=='Average' and method=='All' and len(means)<mx:   #replace the zero values with the average mean/std values of that patient
            features[index,len(means):mx]=np.mean(means)
            features[index,mx+len(stands):]=np.mean(stands)
        
    features=pd.DataFrame(features,columns=names)  #features to a DataFrame
    ages=pd.DataFrame(ages,columns=['Age']) #ages array to DataFrame
    gender=pd.DataFrame(gender,columns=['Gender'])
    extra=pd.merge(ages,gender,right_index=True,left_index=True) #merge Frames
    feats=pd.merge(features,extra,right_index=True,left_index=True)   #merge the Frames
    features_normal=pd.merge(features,extra,right_index=True,left_index=True)  #is not affected by normalization
    y_true=pd.DataFrame(y_true,columns=['Label']) #labels array to DataFrame
    
    if normalize:   #normalization of the feature columns
        scaler = preprocessing.StandardScaler()   #initialize scaler
        feats[feats.columns]=scaler.fit_transform(feats.values)   #normalize the feature values
    
    
    return feats,y_true,mx,features_normal,y_cert

def cross_sc(clf,feat,y_true,y_cert,certainty):
    '''Use cross validation to determine mean and standard deviation of the statistics for the given classifier'''
    n=100   #number of folds to use
    ss = ShuffleSplit(n_splits=n,test_size=0.2)   #make a shuffle split generator
    #make lists to store statistics
    F1s=[]
    AUCs=[]
    kaps=[]
    PPVs=[]
    NPVs=[]
    sensis=[]
    specis=[]
    for train_index, test_index in ss.split(feat):   #loop over each fold
        clf.fit(feat.iloc[train_index],y_true.iloc[train_index])   #fit the classifier to the train data
        pred=clf.predict(feat.iloc[test_index])    #make predictions on the test data
        if certainty:
            pred[np.reshape(y_cert[test_index]==1,len(pred))]=1
            pred[np.reshape(y_cert[test_index]==0,len(pred))]=0
        
        (F1,kap,AUC,PPV,NPV,sensi,speci)=kappa(y_true.iloc[test_index],pred)   #compute statistics
        assert clf.classes_[1]==1   #ensure that the correct predictions are chosen
        #append the statistics
        F1s.append(F1)
        AUCs.append(AUC)
        kaps.append(kap['kappa'])
        PPVs.append(PPV)
        NPVs.append(NPV)
        sensis.append(sensi)
        specis.append(speci)
    return np.mean(kaps),np.std(kaps),np.mean(F1s),np.std(F1s),np.mean(AUCs),np.std(AUCs),np.mean(PPVs),np.std(PPVs),np.mean(NPVs),np.std(NPVs),np.mean(sensis),np.std(sensis),np.mean(specis),np.std(specis)  #calculate the mean and standard deviation of the statistics
    
def classifiers(dframe,predictions,algos,padding,normalize,method,certainty,basic,validation):
    '''Set-up all of the options for classification and calculate the scores'''
    features,y_true,num,features_normal,y_cert=prepare_data(dframe,predictions,padding,normalize,method,certainty)   #acquire the features and labels
    if basic:
        ending=num*2  #if not all features are wished to be taken into account
        features=features[features.columns[0:ending]]
        features_normal=features_normal[features_normal.columns[0:ending]]
        
    if validation:
        pred_val=np.load(path.validation_pred)  #load numpy array
        frame_val=pd.read_excel(path.validation_excel)  #load patient data
        
        X_val,y_val,_,features_normal,y_val_cert=prepare_data(frame_val,pred_val,padding,normalize,method,certainty)   #acquire the features and labels
        if basic:
            ending=num*2  #if not all features are wished to be taken into account
            X_val=X_val[X_val.columns[0:ending]]
            features_normal=features_normal[features_normal.columns[0:ending]]
    #lists for the scores
    AUCs=[]
    AUCs_std=[];
    F1s=[]
    F1_std=[]
    kaps=[]
    kap_std=[]
    PPVs=[]
    PPV_std=[]
    NPVs=[]
    NPV_std=[]
    sensis=[]
    sensi_std=[]
    specis=[]
    speci_std=[]
    classif=[]
    for clas in algos:   #go over all specified options
        disable=False
        if clas=='Decision Tree':
            clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=3,max_depth=3,min_samples_split=10,max_features=None)   #initialize the classifier
        elif clas=='Logistic Regression':
            clf = LogisticRegression(penalty='l2',solver='liblinear',dual=False)    #initialization of the classifier
        elif clas=='Naive Bayes':
            clf = GaussianNB()
        elif clas=='SVM':
            clf = SVC(C=5,gamma='auto')
        #elif clas=='Random Forest':
            #clf = RandomForestClassifier(n_estimators=100,criterion='gini')
        elif clas=='Nearest Neighbour':
            clf = KNeighborsClassifier(n_neighbors=25,weights='distance',algorithm='auto')
        elif clas =='First Only' and method=='All' and validation==False:
            F1,kap,auc,PPV,NPV,sensi,speci=first_only(features_normal,y_true,num)
        elif clas =='First Only' and method=='All' and validation==True:
            F1,kap,auc,PPV,NPV,sensi,speci=first_only(features_normal,y_val,num)
        elif clas =='Averaging' and validation==False:
            F1,kap,auc,PPV,NPV,sensi,speci=averaging(features_normal,y_true,num)
        elif clas =='Averaging' and validation==True:
            F1,kap,auc,PPV,NPV,sensi,speci=averaging(features_normal,y_val,num)
        elif clas =='Majority Vote' and validation==False:
            F1,kap,auc,PPV,NPV,sensi,speci=major(features_normal,y_true,num)
        elif clas =='Majority Vote' and validation==True:
            F1,kap,auc,PPV,NPV,sensi,speci=major(features_normal,y_val,num)
        elif clas=='AdaBoost':
            clf=AdaBoostClassifier(base_estimator=LogisticRegression(penalty='l2',solver='liblinear'),n_estimators=50,learning_rate=0.1)
        #elif clas=='Bagging':
            #clf=BaggingClassifier(KNeighborsClassifier(n_neighbors=20,algorithm='auto'),max_samples=0.5, max_features=0.5)
        #elif clas=='Extra Trees':
            #clf=ExtraTreesClassifier(n_estimators=10)
        #elif clas=='Gradient Boosting':
            #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        elif clas=='Voting classifier':
            clf1=LogisticRegression(penalty='l2',solver='liblinear',dual=False)
            clf2=SVC(C=5,gamma='auto',probability=True)
            clf3=tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=3,max_depth=3,min_samples_split=10,max_features=None)
            clf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('dt', clf3)], voting='soft')
        else:
            disable=True   #to avoid the remaining processes if no correct classifiers are given
        
        if clas=='First Only' or clas=='Averaging' or clas =='Majority Vote':  #these do not have a standard deviation
            ks=0 
            fs=0
            ast=0
            Ps=0
            Ns=0
            sss=0
            sfs=0
        elif validation==False and disable==False:
            kap,ks,F1,fs,auc,ast,PPV,Ps,NPV,Ns,sensi,sss,speci,sfs=cross_sc(clf,features,y_true,y_cert,certainty)    #compute cross-validation score
        elif disable==False and validation==True:
            kap,ks,F1,fs,auc,ast,PPV,Ps,NPV,Ns,sensi,sss,speci,sfs=validation_score(clf,features,y_true,X_val,y_val,y_val_cert,certainty)  
        #add the scores to the list
        if disable==False:
            AUCs.append(auc)
            AUCs_std.append(ast)
            F1s.append(F1)
            F1_std.append(fs)
            kaps.append(kap)
            kap_std.append(ks)
            PPVs.append(PPV)
            PPV_std.append(Ps)
            NPVs.append(NPV)
            NPV_std.append(Ns)
            sensis.append(sensi)
            sensi_std.append(sss)
            specis.append(speci)
            speci_std.append(sfs)
            classif.append(clas)
    return AUCs, AUCs_std, F1s, F1_std, kaps, kap_std,PPVs,PPV_std,NPVs,NPV_std,sensis,sensi_std,specis,speci_std, classif

def perform_pca(dframe,predictions,padding,normalize,method,certainty):
    '''Principal Component analysis'''
    pca=PCA()  #initialize PCA
    feat,y,num,fn,y_cert=prepare_data(dframe,predictions,padding,normalize,method,certainty)  #get features
    pca.fit(feat)   #fit data to PCA 
    #print outcome
    ev=pca.explained_variance_
    print(ev)   
    print(feat.columns)
    
    #calculate and print the ratio of eigenvalues
    ratios=np.zeros((len(ev),1))
    for i in range(0,len(ev)):
        ratios[i]=np.sum(ev[0:i+1])/np.sum(ev)
    print(ratios)
        
    return

def make_bar(frame,frame_basic,save,padding,method):
    '''make the bar plots for each metric with an option to save them'''
    names=frame.columns[1:]  #select the column names wich will have the scores and std's
    plt.rcParams.update(plt.rcParamsDefault)   #go back to default plt settings
    for name in names:  #go over columns
        if 'score' in name:  #if this is the name of a score continue
            metric=name.split(' ')[0]  #define which metric this is
            scores=frame[metric+' score']  #get the scores of this metric
            scoresb=frame_basic[metric+' score']
            stds=frame[metric+' std']   #get the std's of this metric
            stds_bas=frame_basic[metric+' std']
            
            #create the bar plot
            ind = np.arange(len(scores))  # the x locations for the groups
            width = 0.35  # the width of the bars

            f, ax = plt.subplots()
            rectn=ax.bar(ind - width/2,scores,width,yerr=stds,capsize=10,label='Image and Patient Info')
            rect_bas=ax.bar(ind + width/2,scoresb,width,yerr=stds_bas,capsize=10,label='Only Image Info')
            ax.set_ylabel(name)
            ax.set_title(name+' with settings padding: '+padding+', method: '+method)
            ax.set_xticks(ind)
            ax.set_xticklabels(frame['Methods'],rotation='vertical')
            mx=np.max(scores+stds)
            ax.set_ylim([0,mx+0.3])
            ax.legend(loc='upper left')
            
            autolabel(rectn,ax,stds,"center")
            autolabel(rect_bas,ax,stds_bas,"center")
            f.tight_layout()
            plt.show()
            
            if save:      #save the figure if wanted with a unique name to prevent overwriting files
                x=datetime.datetime.now()
                extra='_'.join([metric,str(x.year),str(x.month),str(x.day),str(x.hour),str(x.minute),str(x.second)])
                f.savefig('Bar_plot_'+extra+'.png',bbox_inches='tight')
    return

def autolabel(rects,ax,std, xpos='center'):
    """ Attach a text label above each bar, displaying its height."""

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}  
    offset = {'center': 0, 'right': 1, 'left': -1}

    for index,rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{:0.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height+std[index]),
                    xytext=(offset[xpos], 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom',rotation=90)
    return

def data_visual(dframe,predictions,padding,method):
    '''Visualize the distribution of the data'''
    features,y_true,num,_,_=prepare_data(dframe,predictions,padding,False,method,False)   #acquire the features and labels
    #make a pairplot
    sns.set_style("whitegrid")
    y_true[y_true==0]='Healthy'
    y_true[y_true==1]='Diabetes'
    feat=pd.merge(features,y_true,right_index=True,left_index=True)
    sns.pairplot(feat,hue='Label',size=3)
    plt.show()
    
    #make a heatmap of the correlations
    corr=features.corr()
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
    plt.show()
    
    #show the distribution plost of the means
    if method!='All':
        for i in range(0,num):
            mean=features[features.columns[i]]
            sns.distplot(mean[np.reshape(y_true.values=='Healthy',len(y_true))], hist = False, kde = True,kde_kws = {'shade': True, 'linewidth': 3}, label = 'Healthy')
            sns.distplot(mean[np.reshape(y_true.values=='Diabetes',len(y_true))], hist = False, kde = True,kde_kws = {'shade': True, 'linewidth': 3}, label = 'Diabetes')
            plt.show()
    return

def validation_score(clf,X_train,y_train,X_val,y_val,y_cert_val,certainty,method,padding):
    tree_visual=False   #whether to visualize the DT 
    n=10   #number of models made
    ss = ShuffleSplit(n_splits=n,test_size=0.2)   #make a shuffle split generator
    #clear memory for scores
    F1s=np.zeros((n))
    AUCs=np.zeros((n))
    kaps=np.zeros((n))
    PPVs=np.zeros((n))
    NPVs=np.zeros((n))
    sensis=np.zeros((n))
    specis=np.zeros((n))
    for i,(train_index, test) in enumerate(ss.split(X_train)):  #go over each fod
        clf.fit(X_train.iloc[train_index],y_train.iloc[train_index])   #train on 80% of the training dataset
        pred=clf.predict(X_val)  #predict on the entire validation dataset
    
        if certainty:   #correct mistakes of the labels which were certain
            pred[np.reshape(y_cert_val==1,len(pred))]=1
            pred[np.reshape(y_cert_val==0,len(pred))]=0
        assert clf.classes_[1]==1   #ensure that the correct predictions are chosen
        #calculate the scores and add to the corresponding lists
        (F1,kap,AUC,PPV,NPV,sensi,speci)=kappa(y_val,pred)
        F1s[i]=F1
        AUCs[i]=AUC
        kaps[i]=kap['kappa']
        PPVs[i]=PPV
        NPVs[i]=NPV
        sensis[i]=sensi
        specis[i]=speci
    
    #if the classifier is a Decision Tree and visualization is enabled, save the DT
    string=method+'_'+padding+'.dot' #filename of DT
    if type(clf)==sklearn.tree.tree.DecisionTreeClassifier and tree_visual==True:
        clf.fit(X_train,y_train)   #fit on the entire training dataset
        export_graphviz(clf, out_file=string, feature_names = X_val.columns,class_names = ['Healthy','T2D'],rounded = True, proportion = False, precision = 2, filled = True)
    
    return np.mean(kaps),np.std(kaps),np.mean(F1s),np.std(F1s),np.mean(AUCs),np.std(AUCs),np.mean(PPVs),np.std(PPVs),np.mean(NPVs),np.std(NPVs),np.mean(sensis),np.std(sensis),np.mean(specis),np.std(specis)

class path():
    '''contains the paths the the data'''
    training_pred='train_prediction_30.npy'
    training_excel='labels_trainingset.xlsx'
    validation_pred='val_prediction_30.npy'
    validation_excel='labels_validationset.xlsx'
    
save_bar=False   #option to save bar plots    options: True or False
normalize=True   #option to normalize features    options: True or False
method='Both'  #method to use for data pre-processing    options: 'All' or 'Eyes' or 'Center' or 'Both'
padding='Zero'  #Padding method   options: 'Zero' or 'Average'
certainty=True    #options: True or False
visual=False   #visualze data distributions
Only_kappa=True   #Only show Kappa Score if set to True
validation=True   #Use the validation dataset or not


predictions=np.load(path.training_pred)  #load numpy array
dframe=pd.read_excel(path.training_excel)  #load patient data
if visual:
    data_visual(dframe,predictions,padding,method)
classi=['First Only','Averaging','Majority Vote','Decision Tree','Logistic Regression','Naive Bayes','SVM','Nearest Neighbour','AdaBoost','Voting classifier']
#Classification process with patient information
AUC,Ast,F1,F1st,Kap,Kst,PPV,Ps,NPV,Ns,sensi,Sstd,speci,spst,classif=classifiers(dframe,predictions,classi,padding,normalize,method,certainty,False,validation)
if Only_kappa:
    results_fr=display_results(classif,Kappa_score=Kap,Kappa_std=Kst)   
else:
    results_fr=display_results(classif,F1_score=F1,F1_std=F1st,Kappa_score=Kap,Kappa_std=Kst,AUC_score=AUC,AUC_std=Ast,PPV_score=PPV,PPV_std=Ps,NPV_score=NPV,NPV_std=Ns,sensitivity_score=sensi,sensitivity_std=Sstd,specificity_score=speci,specificity_std=spst)

#Classification process without patient information
AUC,Ast,F1,F1st,Kap,Kst,PPV,Ps,NPV,Ns,sensi,Sstd,speci,spst,classif=classifiers(dframe,predictions,classi,padding,normalize,method,certainty,True,validation)
if Only_kappa:
    results_basic=display_results(classif,Kappa_score=Kap,Kappa_std=Kst)   
else:
    results_basic=display_results(classif,F1_score=F1,F1_std=F1st,Kappa_score=Kap,Kappa_std=Kst,AUC_score=AUC,AUC_std=Ast,PPV_score=PPV,PPV_std=Ps,NPV_score=NPV,NPV_std=Ns,sensitivity_score=sensi,sensitivity_std=Sstd,specificity_score=speci,specificity_std=spst)
if len(results_fr)>0:  #if any selected classifiers have been used, display results
    make_bar(results_fr,results_basic,save_bar,padding,method)
else:
    print('Please select a classifier programmed in this code')

end = time.time()
print('time: '+ '{:0.1f}'.format(end - start)+' sec')