# Importing Required Libraries for the project
# Importing numpy for managing operations in array
# Importing Pandas for data manupulation
# Importing tkinter for Frontend GUI


import pandas as panda

from tkinter import *

import numpy as nump




list11=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

diseases=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

list22=[]
for x in range(0,len(list11)):
    list22.append(0)

# disease prediction - test dataset dataframe
dataframe=panda.read_csv("Training.csv")



dataframe.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X1= dataframe[list11]

y1 = dataframe[["prognosis"]]

nump.ravel(y1)

# TRAINING DATA 
train=panda.read_csv("Testing.csv")
train.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

Xtest= train[list11]

ytest = train[["prognosis"]]

nump.ravel(ytest)


# Module 1: Decision tree Classifier

def DecTree():

    #importing required modules
    from sklearn import tree
    from sklearn.metrics import accuracy_score

    clf2 = tree.DecisionTreeClassifier()
    # model of the tree
    clf2 = clf2.fit(X1,y1)

    # calculation of the accuracy of the model
    ypred=clf2.predict(Xtest)
    print(accuracy_score(ytest, ypred))
    print(accuracy_score(ytest, ypred,normalize=False))


    psymp = [Symp1.get(),Symp2.get(),Symp3.get(),Symp4.get(),Symp5.get()]

    for ko in range(0,len(list11)):
    
        for zen in psymp:
            if(zen==list11[ko]):
                list22[ko]=1


    it = [list22]
    
    prediction = clf2.predict(it)
    
    predictd=prediction[0]


    hh='no'
    for ah in range(0,len(diseases)):
        if(predictd == ah):
            hh='yes'
            break



    if (hh=='yes'):
        tf1.delete("1.0", END)
        tf1.insert(END, diseases[ah])
    else:
        tf1.delete("1.0", END)
        tf1.insert(END, "Nothing")


# Module 2: RandomForest Classifier

def rand():

    #Importing required modules
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    clf5 = RandomForestClassifier()
    clf5 = clf5.fit(X1,nump.ravel(y1))


    # calculating accuracy

    ypred=clf5.predict(Xtest)
    print(accuracy_score(ytest, ypred))
    print(accuracy_score(ytest, ypred,normalize=False))


    psymp = [Symp1.get(),Symp2.get(),Symp3.get(),Symp4.get(),Symp5.get()]

    for ko in range(0,len(list11)):
        for zen in psymp:
            if(zen==list11[ko]):
                list22[ko]=1

    it = [list22]
    
    prediction = clf5.predict(it)
    
    predictd=prediction[0]



    hh='no'
    for ah in range(0,len(diseases)):
        if(predictd == ah):
            hh='yes'
            break


    if (hh=='yes'):
        tf2.delete("1.0", END)
        tf2.insert(END, diseases[ah])
    else:
        tf2.delete("1.0", END)
        tf2.insert(END, "Nothing")


        

# Module 3: NaiveBayes


def Naveb():

    # importing required modules
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import GaussianNB
    
    ganb = GaussianNB()
    ganb=ganb.fit(X1,nump.ravel(y1))


    # calculating accuracy
   
    ypred=ganb.predict(Xtest)
    print(accuracy_score(ytest, ypred))
    print(accuracy_score(ytest, ypred,normalize=False))


    psymp= [Symp1.get(),Symp2.get(),Symp3.get(),Symp4.get(),Symp5.get()]
    for ko in range(0,len(list11)):
        for z in psymp:
            if(z==list11[ko]):
                list22[ko]=1

    it = [list22]
    
    prediction = ganb.predict(it)

    predictd=prediction[0]

    hh='no'
    for ah in range(0,len(diseases)):
        if(predictd == ah):
            hh='yes'
            break

    if (hh=='yes'):
        tf3.delete("1.0", END)
        tf3.insert(END, diseases[ah])
    else:
        tf3.delete("1.0", END)
        tf3.insert(END, "Nothing")

# Module 4: using TKInterface for Graphical user interface

cute = Tk()
cute.configure(background='#483D8B')

# var for entries
Symp1 = StringVar()
Symp1.set(None)
Symp2 = StringVar()
Symp2.set(None)
Symp3 = StringVar()
Symp3.set(None)
Symp4 = StringVar()
Symp4.set(None)
Symp5 = StringVar()
Symp5.set(None)
Name = StringVar()

# Heading

wall2 = Label(cute, justify=LEFT, text="Mini Project", fg="white", bg="#483D8B")
wall2.config(font=("Forte", 30))
wall2.grid(row=1, column=0, columnspan=2, padx=100)
wall2 = Label(cute, justify=LEFT, text="Disease Predictor", fg="white", bg="#483D8B")
wall2.config(font=("Forte", 30))
wall2.grid(row=2, column=0, columnspan=2, padx=100)

# outer
NameLable = Label(cute, text="Patient's Name", fg="yellow", bg="black")
NameLable.grid(row=6, column=1, pady=15, sticky=W)


Symp1Lb = Label(cute, text="Symptom 1", fg="#FFFF00", bg="#000000")
Symp1Lb.grid(row=8, column=1, pady=10, sticky=W)

Symp2Lb = Label(cute, text="Symptom 2", fg="#FFFF00", bg="#000000")
Symp2Lb.grid(row=9, column=1, pady=10, sticky=W)

Symp3Lb = Label(cute, text="Symptom 3", fg="#FFFF00", bg="#000000")
Symp3Lb.grid(row=10, column=1, pady=10, sticky=W)

Symp4Lb = Label(cute, text="Symptom 4", fg="#FFFF00", bg="#000000")
Symp4Lb.grid(row=11, column=1, pady=10, sticky=W)

Symp5Lb = Label(cute, text="Symptom 5", fg="#FFFF00", bg="#000000")
Symp5Lb.grid(row=12, column=1, pady=10, sticky=W)


layLb = Label(cute, text="Decision Tree", fg="white", bg="red")
layLb.grid(row=16, column=1, pady=10,sticky=W)

desLb = Label(cute, text="Random Forest", fg="white", bg="red")
desLb.grid(row=18, column=1, pady=10, sticky=W)

navfLb = Label(cute, text="Naïve Bayes", fg="white", bg="red")
navfLb.grid(row=20, column=1, pady=10, sticky=W)

# root portion for ent-ry

OPT = sorted(list11)

NameOfEntry = Entry(cute, textvariable=Name)
NameOfEntry.grid(row=6, column=1)

S1En = OptionMenu(cute, Symp1,*OPT)
S1En.grid(row=8, column=1)

S2En = OptionMenu(cute, Symp2,*OPT)
S2En.grid(row=9, column=1)

S3En = OptionMenu(cute, Symp3,*OPT)
S3En.grid(row=10, column=1)

S4En = OptionMenu(cute, Symp4,*OPT)
S4En.grid(row=11, column=1)

S5En = OptionMenu(cute, Symp5,*OPT)
S5En.grid(row=12, column=1)


dstree = Button(cute, text="DecisionTree Classifier", command=DecTree,bg="#556b2f",fg="#FFFF00")
dstree.grid(row=9, column=2,padx=11)

rndmf = Button(cute, text="RandomForest Classifier", command=rand,bg="#556b2f",fg="#FFFF00")
rndmf.grid(row=10, column=2,padx=11)

lrnb = Button(cute, text="Naïve Bayes", command=Naveb,bg="#556b2f",fg="#FFFF00")
lrnb.grid(row=11, column=2,padx=11)

#project ans outlet

tf1 = Text(cute, height=1, width=20,bg="#7ca9f4",fg="#000000")
tf1.grid(row=16, column=1, padx=10)

tf2 = Text(cute, height=1, width=20,bg="#7ca9f4",fg="#000000")
tf2.grid(row=18, column=1 , padx=10)

tf3 = Text(cute, height=1, width=20,bg="#7ca9f4",fg="#000000")
tf3.grid(row=20, column=1 , padx=10)


cute.mainloop()
