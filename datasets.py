import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#### ADD Dataset TODO
# 1.
# write get_fx
# 2.
# define sensitive_features_dict
# 3.
#

german_sensitive_features_dict = {"gender": 8, "age": 12}


# german_sensitive_features_dict = {"gender": 8, "age": 12, "foreign_work": -1}


def get_german(sensitive_feature_name, filepath=None, remove_z=False, **kwargs):
    if filepath is None:
        filepath = "/home/btd26/datasets/german/german.data-numeric"
    df = pd.read_csv(filepath, header=None, delim_whitespace=True)

    # change label to 0/1
    cols = list(df)
    label_idx = len(cols) - 1
    df[label_idx] = df[label_idx].map({2: 0, 1: 1})

    M = df.values
    X = M[:, :-1]
    y = M[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    z_idx = get_z_idx(sensitive_feature_name, german_sensitive_features_dict)
    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z=remove_z)
    return Xtr, Xts, ytr, yts, Ztr, Zts


def get_z_idx(sensitive_feature_name, sensitive_features_dict):
    z_idx = sensitive_features_dict.get(sensitive_feature_name, None)
    if z_idx is None:
        print("Feature {} not recognized".format(sensitive_feature_name))
        z_idx = 0
    return z_idx


adult_sensitive_features_dict = {"gender": 9, "age": 0, "race": 8}
adult_column_names = ['age', 'workclass', 'fnlwgt', 'education',
                      'education-num', 'marital-status', 'occupation', 'relationship',
                      'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                      'native-country', 'income-per-year']


def get_adult(sensitive_feature_name, scale=True, remove_z=False, verbose=0,**kwargs):
    import os
    file_path = "/home/btd26/datasets/adult/"

    if scale:
        file_name = "adult.npz"
        arr_holder = np.load(os.path.join(file_path, file_name))
        fit_scale = arr_holder[arr_holder.files[0]]

        M = fit_scale
    else:
        file_name = "adult.data"
        df = pd.read_csv(os.path.join(file_path, file_name), sep=",", header=None)
        categorical_columns = []
        for c in df.columns:
            if df[c].dtype is np.dtype('O'):
                categorical_columns.append(c)
        from collections import defaultdict
        d_label = defaultdict(LabelEncoder)
        fit = df.apply(lambda x: d_label[x.name].fit_transform(x) if x.dtype == np.dtype('O') else x)
        M = fit.values

        # TODO persistent & dynamic reading
    X = M[:, :-1]
    y = M[:, -1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if verbose:
        # print shapes
        for x in [X_train, X_test, y_train, y_test]:
            print(x.shape)
    z_idx = get_z_idx(sensitive_feature_name, adult_sensitive_features_dict)
    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z)
    return Xtr, Xts, ytr, yts, Ztr, Zts


bank_sensitive_features_dict = {"marital": 2, "age": 0}


def get_bank(sensitive_feature_name, remove_z=False, **kwargs):
    # assume
    # 0 age
    # 2 marital

    z_idx = get_z_idx(sensitive_feature_name, bank_sensitive_features_dict)

    file_path = "/home/btd26/datasets/bank/"
    file_name = "bank.npz"
    file_add = os.path.join(file_path, file_name)
    if os.path.exists(file_add):
        arr_holder = np.load(os.path.join(file_path, file_name))
        fit_scale = arr_holder[arr_holder.files[0]]
    else:
        filepath = "/home/btd26/datasets/bank/bank-additional/bank-additional-full.csv"
        df = pd.read_csv(filepath, sep=";")
        categorical_columns = []
        for c in df.columns:
            if df[c].dtype is np.dtype('O'):
                categorical_columns.append(c)
        from collections import defaultdict
        d_label = defaultdict(LabelEncoder)
        fit = df.apply(lambda x: d_label[x.name].fit_transform(x) if x.dtype == np.dtype('O') else x)
        # scale
        d_scale = defaultdict(StandardScaler)
        scaler = StandardScaler()
        feature_columns = fit.columns[:-1]
        fit_scale = scaler.fit_transform(fit[feature_columns])
        fit_scale = np.concatenate([fit_scale, fit.iloc[:, -1].values.reshape(-1, 1)], axis=1)

    M = fit_scale
    # M = fit.values
    X = M[:, :-1]
    y = M[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z=remove_z)

    return Xtr, Xts, ytr, yts, Ztr, Zts


compas_column_names = [['sex',
                        'age',
                        'race',
                        'juv_fel_count',
                        'juv_misd_count',
                        'juv_other_count',
                        'priors_count',
                        'age_cat=25 - 45',
                        'age_cat=Greater than 45',
                        'age_cat=Less than 25',
                        'c_charge_degree=F',
                        'c_charge_degree=M',
                        'c_charge_desc=Abuse Without Great Harm',
                        'c_charge_desc=Agg Abuse Elderlly/Disabled Adult',
                        'c_charge_desc=Agg Assault W/int Com Fel Dome',
                        'c_charge_desc=Agg Battery Grt/Bod/Harm',
                        'c_charge_desc=Agg Fleeing and Eluding',
                        'c_charge_desc=Agg Fleeing/Eluding High Speed',
                        'c_charge_desc=Aggr Child Abuse-Torture,Punish',
                        'c_charge_desc=Aggrav Battery w/Deadly Weapon',
                        'c_charge_desc=Aggrav Child Abuse-Agg Battery',
                        'c_charge_desc=Aggrav Child Abuse-Causes Harm',
                        'c_charge_desc=Aggrav Stalking After Injunctn',
                        'c_charge_desc=Aggravated Assault',
                        'c_charge_desc=Aggravated Assault W/Dead Weap',
                        'c_charge_desc=Aggravated Assault W/dead Weap',
                        'c_charge_desc=Aggravated Assault W/o Firearm',
                        'c_charge_desc=Aggravated Assault w/Firearm',
                        'c_charge_desc=Aggravated Battery',
                        'c_charge_desc=Aggravated Battery (Firearm)',
                        'c_charge_desc=Aggravated Battery (Firearm/Actual Possession)',
                        'c_charge_desc=Aggravated Battery / Pregnant',
                        'c_charge_desc=Aggravated Battery On 65/Older',
                        'c_charge_desc=Aide/Abet Prostitution Lewdness',
                        'c_charge_desc=Aiding Escape',
                        'c_charge_desc=Alcoholic Beverage Violation-FL',
                        'c_charge_desc=Armed Trafficking in Cannabis',
                        'c_charge_desc=Arson in the First Degree',
                        'c_charge_desc=Assault',
                        'c_charge_desc=Assault Law Enforcement Officer',
                        'c_charge_desc=Att Burgl Conv Occp',
                        'c_charge_desc=Att Burgl Struc/Conv Dwel/Occp',
                        'c_charge_desc=Att Burgl Unoccupied Dwel',
                        'c_charge_desc=Att Tamper w/Physical Evidence',
                        'c_charge_desc=Attempt Armed Burglary Dwell',
                        'c_charge_desc=Attempted Burg/Convey/Unocc',
                        'c_charge_desc=Attempted Burg/struct/unocc',
                        'c_charge_desc=Attempted Deliv Control Subst',
                        'c_charge_desc=Attempted Robbery  No Weapon',
                        'c_charge_desc=Attempted Robbery  Weapon',
                        'c_charge_desc=Battery',
                        'c_charge_desc=Battery Emergency Care Provide',
                        'c_charge_desc=Battery On A Person Over 65',
                        'c_charge_desc=Battery On Fire Fighter',
                        'c_charge_desc=Battery On Parking Enfor Speci',
                        'c_charge_desc=Battery Spouse Or Girlfriend',
                        'c_charge_desc=Battery on Law Enforc Officer',
                        'c_charge_desc=Battery on a Person Over 65',
                        'c_charge_desc=Bribery Athletic Contests',
                        'c_charge_desc=Burgl Dwel/Struct/Convey Armed',
                        'c_charge_desc=Burglary Assault/Battery Armed',
                        'c_charge_desc=Burglary Conveyance Armed',
                        'c_charge_desc=Burglary Conveyance Assault/Bat',
                        'c_charge_desc=Burglary Conveyance Occupied',
                        'c_charge_desc=Burglary Conveyance Unoccup',
                        'c_charge_desc=Burglary Dwelling Armed',
                        'c_charge_desc=Burglary Dwelling Assault/Batt',
                        'c_charge_desc=Burglary Dwelling Occupied',
                        'c_charge_desc=Burglary Structure Assault/Batt',
                        'c_charge_desc=Burglary Structure Occupied',
                        'c_charge_desc=Burglary Structure Unoccup',
                        'c_charge_desc=Burglary Unoccupied Dwelling',
                        'c_charge_desc=Burglary With Assault/battery',
                        'c_charge_desc=Carjacking w/o Deadly Weapon',
                        'c_charge_desc=Carjacking with a Firearm',
                        'c_charge_desc=Carry Open/Uncov Bev In Pub',
                        'c_charge_desc=Carrying A Concealed Weapon',
                        'c_charge_desc=Carrying Concealed Firearm',
                        'c_charge_desc=Cash Item w/Intent to Defraud',
                        'c_charge_desc=Child Abuse',
                        'c_charge_desc=Computer Pornography',
                        'c_charge_desc=Consp Traff Oxycodone  4g><14g',
                        'c_charge_desc=Conspiracy Dealing Stolen Prop',
                        'c_charge_desc=Consume Alcoholic Bev Pub',
                        'c_charge_desc=Contradict Statement',
                        'c_charge_desc=Contribute Delinquency Of A Minor',
                        'c_charge_desc=Corrupt Public Servant',
                        'c_charge_desc=Counterfeit Lic Plates/Sticker',
                        'c_charge_desc=Crim Attempt/Solic/Consp',
                        'c_charge_desc=Crim Use of Personal ID Info',
                        'c_charge_desc=Crimin Mischief Damage $1000+',
                        'c_charge_desc=Criminal Mischief',
                        'c_charge_desc=Criminal Mischief Damage <$200',
                        'c_charge_desc=Criminal Mischief>$200<$1000',
                        'c_charge_desc=Crlty Twrd Child Urge Oth Act',
                        'c_charge_desc=Cruelty Toward Child',
                        'c_charge_desc=Cruelty to Animals',
                        'c_charge_desc=Culpable Negligence',
                        'c_charge_desc=D.U.I. Serious Bodily Injury',
                        'c_charge_desc=DOC/Cause Public Danger',
                        'c_charge_desc=DUI - Enhanced',
                        'c_charge_desc=DUI - Property Damage/Personal Injury',
                        'c_charge_desc=DUI Blood Alcohol Above 0.20',
                        'c_charge_desc=DUI Level 0.15 Or Minor In Veh',
                        'c_charge_desc=DUI Property Damage/Injury',
                        'c_charge_desc=DUI- Enhanced',
                        'c_charge_desc=DUI/Property Damage/Persnl Inj',
                        'c_charge_desc=DWI w/Inj Susp Lic / Habit Off',
                        'c_charge_desc=DWLS Canceled Disqul 1st Off',
                        'c_charge_desc=DWLS Susp/Cancel Revoked',
                        'c_charge_desc=Dealing in Stolen Property',
                        'c_charge_desc=Defrauding Innkeeper',
                        'c_charge_desc=Defrauding Innkeeper $300/More',
                        'c_charge_desc=Del 3,4 Methylenedioxymethcath',
                        'c_charge_desc=Del Cannabis At/Near Park',
                        'c_charge_desc=Del Cannabis For Consideration',
                        'c_charge_desc=Del of JWH-250 2-Methox 1-Pentyl',
                        'c_charge_desc=Deliver 3,4 Methylenediox',
                        'c_charge_desc=Deliver Alprazolam',
                        'c_charge_desc=Deliver Cannabis',
                        'c_charge_desc=Deliver Cannabis 1000FTSch',
                        'c_charge_desc=Deliver Cocaine',
                        'c_charge_desc=Deliver Cocaine 1000FT Church',
                        'c_charge_desc=Deliver Cocaine 1000FT Park',
                        'c_charge_desc=Deliver Cocaine 1000FT School',
                        'c_charge_desc=Deliver Cocaine 1000FT Store',
                        'c_charge_desc=Delivery Of Drug Paraphernalia',
                        'c_charge_desc=Delivery of 5-Fluoro PB-22',
                        'c_charge_desc=Delivery of Heroin',
                        'c_charge_desc=Depriv LEO of Protect/Communic',
                        'c_charge_desc=Disorderly Conduct',
                        'c_charge_desc=Disorderly Intoxication',
                        'c_charge_desc=Disrupting School Function',
                        'c_charge_desc=Drivg While Lic Suspd/Revk/Can',
                        'c_charge_desc=Driving License Suspended',
                        'c_charge_desc=Driving Under The Influence',
                        'c_charge_desc=Driving While License Revoked',
                        'c_charge_desc=Escape',
                        'c_charge_desc=Exhibition Weapon School Prop',
                        'c_charge_desc=Expired DL More Than 6 Months',
                        'c_charge_desc=Exposes Culpable Negligence',
                        'c_charge_desc=Extradition/Defendants',
                        'c_charge_desc=Fabricating Physical Evidence',
                        'c_charge_desc=Fail Register Vehicle',
                        'c_charge_desc=Fail Sex Offend Report Bylaw',
                        'c_charge_desc=Fail To Obey Police Officer',
                        'c_charge_desc=Fail To Redeliv Hire/Leas Prop',
                        'c_charge_desc=Failure To Pay Taxi Cab Charge',
                        'c_charge_desc=Failure To Return Hired Vehicle',
                        'c_charge_desc=False 911 Call',
                        'c_charge_desc=False Bomb Report',
                        'c_charge_desc=False Imprisonment',
                        'c_charge_desc=False Info LEO During Invest',
                        'c_charge_desc=False Motor Veh Insurance Card',
                        'c_charge_desc=False Name By Person Arrest',
                        'c_charge_desc=False Ownership Info/Pawn Item',
                        'c_charge_desc=Falsely Impersonating Officer',
                        'c_charge_desc=Fel Drive License Perm Revoke',
                        'c_charge_desc=Felon in Pos of Firearm or Amm',
                        'c_charge_desc=Felony Batt(Great Bodily Harm)',
                        'c_charge_desc=Felony Battery',
                        'c_charge_desc=Felony Battery (Dom Strang)',
                        'c_charge_desc=Felony Battery w/Prior Convict',
                        'c_charge_desc=Felony Committing Prostitution',
                        'c_charge_desc=Felony DUI (level 3)',
                        'c_charge_desc=Felony DUI - Enhanced',
                        'c_charge_desc=Felony Driving While Lic Suspd',
                        'c_charge_desc=Felony Petit Theft',
                        'c_charge_desc=Felony/Driving Under Influence',
                        'c_charge_desc=Fighting/Baiting Animals',
                        'c_charge_desc=Fleeing Or Attmp Eluding A Leo',
                        'c_charge_desc=Fleeing or Eluding a LEO',
                        'c_charge_desc=Forging Bank Bills/Promis Note',
                        'c_charge_desc=Fraudulent Use of Credit Card',
                        'c_charge_desc=Grand Theft (Motor Vehicle)',
                        'c_charge_desc=Grand Theft Dwell Property',
                        'c_charge_desc=Grand Theft Firearm',
                        'c_charge_desc=Grand Theft in the 1st Degree',
                        'c_charge_desc=Grand Theft in the 3rd Degree',
                        'c_charge_desc=Grand Theft of a Fire Extinquisher',
                        'c_charge_desc=Grand Theft of the 2nd Degree',
                        'c_charge_desc=Grand Theft on 65 Yr or Older',
                        'c_charge_desc=Harass Witness/Victm/Informnt',
                        'c_charge_desc=Harm Public Servant Or Family',
                        'c_charge_desc=Hiring with Intent to Defraud',
                        'c_charge_desc=Imperson Public Officer or Emplyee',
                        'c_charge_desc=Interfere W/Traf Cont Dev RR',
                        'c_charge_desc=Interference with Custody',
                        'c_charge_desc=Intoxicated/Safety Of Another',
                        'c_charge_desc=Introduce Contraband Into Jail',
                        'c_charge_desc=Issuing a Worthless Draft',
                        'c_charge_desc=Kidnapping / Domestic Violence',
                        'c_charge_desc=Lease For Purpose Trafficking',
                        'c_charge_desc=Leave Acc/Attend Veh/More $50',
                        'c_charge_desc=Leave Accd/Attend Veh/Less $50',
                        'c_charge_desc=Leaving Acc/Unattended Veh',
                        'c_charge_desc=Leaving the Scene of Accident',
                        'c_charge_desc=Lewd Act Presence Child 16-',
                        'c_charge_desc=Lewd or Lascivious Molestation',
                        'c_charge_desc=Lewd/Lasc Battery Pers 12+/<16',
                        'c_charge_desc=Lewd/Lasc Exhib Presence <16yr',
                        'c_charge_desc=Lewd/Lasciv Molest Elder Persn',
                        'c_charge_desc=Lewdness Violation',
                        'c_charge_desc=License Suspended Revoked',
                        'c_charge_desc=Littering',
                        'c_charge_desc=Live on Earnings of Prostitute',
                        'c_charge_desc=Lve/Scen/Acc/Veh/Prop/Damage',
                        'c_charge_desc=Manage Busn W/O City Occup Lic',
                        'c_charge_desc=Manslaughter W/Weapon/Firearm',
                        'c_charge_desc=Manufacture Cannabis',
                        'c_charge_desc=Misuse Of 911 Or E911 System',
                        'c_charge_desc=Money Launder 100K or More Dols',
                        'c_charge_desc=Murder In 2nd Degree W/firearm',
                        'c_charge_desc=Murder in the First Degree',
                        'c_charge_desc=Neglect Child / Bodily Harm',
                        'c_charge_desc=Neglect Child / No Bodily Harm',
                        'c_charge_desc=Neglect/Abuse Elderly Person',
                        'c_charge_desc=Obstruct Fire Equipment',
                        'c_charge_desc=Obstruct Officer W/Violence',
                        'c_charge_desc=Obtain Control Substance By Fraud',
                        'c_charge_desc=Offer Agree Secure For Lewd Act',
                        'c_charge_desc=Offer Agree Secure/Lewd Act',
                        'c_charge_desc=Offn Against Intellectual Prop',
                        'c_charge_desc=Open Carrying Of Weapon',
                        'c_charge_desc=Oper Motorcycle W/O Valid DL',
                        'c_charge_desc=Operating W/O Valid License',
                        'c_charge_desc=Opert With Susp DL 2nd Offens',
                        'c_charge_desc=PL/Unlaw Use Credit Card',
                        'c_charge_desc=Petit Theft',
                        'c_charge_desc=Petit Theft $100- $300',
                        'c_charge_desc=Pos Cannabis For Consideration',
                        'c_charge_desc=Pos Cannabis W/Intent Sel/Del',
                        'c_charge_desc=Pos Methylenedioxymethcath W/I/D/S',
                        'c_charge_desc=Poss 3,4 MDMA (Ecstasy)',
                        'c_charge_desc=Poss Alprazolam W/int Sell/Del',
                        'c_charge_desc=Poss Anti-Shoplifting Device',
                        'c_charge_desc=Poss Cntrft Contr Sub w/Intent',
                        'c_charge_desc=Poss Cocaine/Intent To Del/Sel',
                        'c_charge_desc=Poss Contr Subst W/o Prescript',
                        'c_charge_desc=Poss Counterfeit Payment Inst',
                        'c_charge_desc=Poss Drugs W/O A Prescription',
                        'c_charge_desc=Poss F/Arm Delinq',
                        'c_charge_desc=Poss Firearm W/Altered ID#',
                        'c_charge_desc=Poss Meth/Diox/Meth/Amp (MDMA)',
                        'c_charge_desc=Poss Of 1,4-Butanediol',
                        'c_charge_desc=Poss Of Controlled Substance',
                        'c_charge_desc=Poss Of RX Without RX',
                        'c_charge_desc=Poss Oxycodone W/Int/Sell/Del',
                        'c_charge_desc=Poss Pyrrolidinobutiophenone',
                        'c_charge_desc=Poss Pyrrolidinovalerophenone',
                        'c_charge_desc=Poss Pyrrolidinovalerophenone W/I/D/S',
                        'c_charge_desc=Poss Similitude of Drivers Lic',
                        'c_charge_desc=Poss Tetrahydrocannabinols',
                        'c_charge_desc=Poss Unlaw Issue Driver Licenc',
                        'c_charge_desc=Poss Unlaw Issue Id',
                        'c_charge_desc=Poss Wep Conv Felon',
                        'c_charge_desc=Poss of Cocaine W/I/D/S 1000FT Park',
                        'c_charge_desc=Poss of Firearm by Convic Felo',
                        'c_charge_desc=Poss of Methylethcathinone',
                        'c_charge_desc=Poss/Sell/Del Cocaine 1000FT Sch',
                        'c_charge_desc=Poss/Sell/Del/Man Amobarbital',
                        'c_charge_desc=Poss/pur/sell/deliver Cocaine',
                        'c_charge_desc=Poss3,4 Methylenedioxymethcath',
                        'c_charge_desc=Posses/Disply Susp/Revk/Frd DL',
                        'c_charge_desc=Possess Cannabis 1000FTSch',
                        'c_charge_desc=Possess Cannabis/20 Grams Or Less',
                        'c_charge_desc=Possess Controlled Substance',
                        'c_charge_desc=Possess Countrfeit Credit Card',
                        'c_charge_desc=Possess Drug Paraphernalia',
                        'c_charge_desc=Possess Mot Veh W/Alt Vin #',
                        'c_charge_desc=Possess Tobacco Product Under 18',
                        'c_charge_desc=Possess Weapon On School Prop',
                        'c_charge_desc=Possess w/I/Utter Forged Bills',
                        'c_charge_desc=Possession Burglary Tools',
                        'c_charge_desc=Possession Child Pornography',
                        'c_charge_desc=Possession Firearm School Prop',
                        'c_charge_desc=Possession Of 3,4Methylenediox',
                        'c_charge_desc=Possession Of Alprazolam',
                        'c_charge_desc=Possession Of Amphetamine',
                        'c_charge_desc=Possession Of Anabolic Steroid',
                        'c_charge_desc=Possession Of Buprenorphine',
                        'c_charge_desc=Possession Of Carisoprodol',
                        'c_charge_desc=Possession Of Clonazepam',
                        'c_charge_desc=Possession Of Cocaine',
                        'c_charge_desc=Possession Of Diazepam',
                        'c_charge_desc=Possession Of Fentanyl',
                        'c_charge_desc=Possession Of Heroin',
                        'c_charge_desc=Possession Of Methamphetamine',
                        'c_charge_desc=Possession Of Paraphernalia',
                        'c_charge_desc=Possession Of Phentermine',
                        'c_charge_desc=Possession of Alcohol Under 21',
                        'c_charge_desc=Possession of Benzylpiperazine',
                        'c_charge_desc=Possession of Butylone',
                        'c_charge_desc=Possession of Cannabis',
                        'c_charge_desc=Possession of Cocaine',
                        'c_charge_desc=Possession of Codeine',
                        'c_charge_desc=Possession of Ethylone',
                        'c_charge_desc=Possession of Hydrocodone',
                        'c_charge_desc=Possession of Hydromorphone',
                        'c_charge_desc=Possession of LSD',
                        'c_charge_desc=Possession of Methadone',
                        'c_charge_desc=Possession of Morphine',
                        'c_charge_desc=Possession of Oxycodone',
                        'c_charge_desc=Possession of XLR11',
                        'c_charge_desc=Principal In The First Degree',
                        'c_charge_desc=Prostitution',
                        'c_charge_desc=Prostitution/Lewd Act Assignation',
                        'c_charge_desc=Prostitution/Lewdness/Assign',
                        'c_charge_desc=Prowling/Loitering',
                        'c_charge_desc=Purchase Cannabis',
                        'c_charge_desc=Purchase/P/W/Int Cannabis',
                        'c_charge_desc=Reckless Driving',
                        'c_charge_desc=Refuse Submit Blood/Breath Test',
                        'c_charge_desc=Refuse to Supply DNA Sample',
                        'c_charge_desc=Resist Officer w/Violence',
                        'c_charge_desc=Resist/Obstruct W/O Violence',
                        'c_charge_desc=Retail Theft $300 1st Offense',
                        'c_charge_desc=Retail Theft $300 2nd Offense',
                        'c_charge_desc=Ride Tri-Rail Without Paying',
                        'c_charge_desc=Robbery / No Weapon',
                        'c_charge_desc=Robbery / Weapon',
                        'c_charge_desc=Robbery Sudd Snatch No Weapon',
                        'c_charge_desc=Robbery W/Deadly Weapon',
                        'c_charge_desc=Robbery W/Firearm',
                        'c_charge_desc=Sale/Del Cannabis At/Near Scho',
                        'c_charge_desc=Sale/Del Counterfeit Cont Subs',
                        'c_charge_desc=Sel/Pur/Mfr/Del Control Substa',
                        'c_charge_desc=Sell or Offer for Sale Counterfeit Goods',
                        'c_charge_desc=Sell/Man/Del Pos/w/int Heroin',
                        'c_charge_desc=Sex Batt Faml/Cust Vict 12-17Y',
                        'c_charge_desc=Sex Battery Deft 18+/Vict 11-',
                        'c_charge_desc=Sex Offender Fail Comply W/Law',
                        'c_charge_desc=Sexual Battery / Vict 12 Yrs +',
                        'c_charge_desc=Sexual Performance by a Child',
                        'c_charge_desc=Shoot In Occupied Dwell',
                        'c_charge_desc=Shoot Into Vehicle',
                        'c_charge_desc=Simulation of Legal Process',
                        'c_charge_desc=Solic to Commit Battery',
                        'c_charge_desc=Solicit Deliver Cocaine',
                        'c_charge_desc=Solicit Purchase Cocaine',
                        'c_charge_desc=Solicit To Deliver Cocaine',
                        'c_charge_desc=Solicitation On Felony 3 Deg',
                        'c_charge_desc=Soliciting For Prostitution',
                        'c_charge_desc=Sound Articles Over 100',
                        'c_charge_desc=Stalking',
                        'c_charge_desc=Stalking (Aggravated)',
                        'c_charge_desc=Strong Armed  Robbery',
                        'c_charge_desc=Structuring Transactions',
                        'c_charge_desc=Susp Drivers Lic 1st Offense',
                        'c_charge_desc=Tamper With Victim',
                        'c_charge_desc=Tamper With Witness',
                        'c_charge_desc=Tamper With Witness/Victim/CI',
                        'c_charge_desc=Tampering With Physical Evidence',
                        'c_charge_desc=Tampering with a Victim',
                        'c_charge_desc=Theft/To Deprive',
                        'c_charge_desc=Threat Public Servant',
                        'c_charge_desc=Throw Deadly Missile Into Veh',
                        'c_charge_desc=Throw In Occupied Dwell',
                        'c_charge_desc=Throw Missile Into Pub/Priv Dw',
                        'c_charge_desc=Traff In Cocaine <400g>150 Kil',
                        'c_charge_desc=Traffic Counterfeit Cred Cards',
                        'c_charge_desc=Traffick Amphetamine 28g><200g',
                        'c_charge_desc=Traffick Oxycodone     4g><14g',
                        'c_charge_desc=Trans/Harm/Material to a Minor',
                        'c_charge_desc=Trespass On School Grounds',
                        'c_charge_desc=Trespass Other Struct/Conve',
                        'c_charge_desc=Trespass Private Property',
                        'c_charge_desc=Trespass Property w/Dang Weap',
                        'c_charge_desc=Trespass Struct/Conveyance',
                        'c_charge_desc=Trespass Structure w/Dang Weap',
                        'c_charge_desc=Trespass Structure/Conveyance',
                        'c_charge_desc=Trespassing/Construction Site',
                        'c_charge_desc=Tresspass Struct/Conveyance',
                        'c_charge_desc=Tresspass in Structure or Conveyance',
                        'c_charge_desc=Unauth C/P/S Sounds>1000/Audio',
                        'c_charge_desc=Unauth Poss ID Card or DL',
                        'c_charge_desc=Unauthorized Interf w/Railroad',
                        'c_charge_desc=Unl/Disturb Education/Instui',
                        'c_charge_desc=Unlaw Lic Use/Disply Of Others',
                        'c_charge_desc=Unlaw LicTag/Sticker Attach',
                        'c_charge_desc=Unlaw Use False Name/Identity',
                        'c_charge_desc=Unlawful Conveyance of Fuel',
                        'c_charge_desc=Unlicensed Telemarketing',
                        'c_charge_desc=Use Computer for Child Exploit',
                        'c_charge_desc=Use Of 2 Way Device To Fac Fel',
                        'c_charge_desc=Use Scanning Device to Defraud',
                        'c_charge_desc=Use of Anti-Shoplifting Device',
                        'c_charge_desc=Uttering Forged Bills',
                        'c_charge_desc=Uttering Forged Credit Card',
                        'c_charge_desc=Uttering Worthless Check +$150',
                        'c_charge_desc=Uttering a Forged Instrument',
                        'c_charge_desc=Video Voyeur-<24Y on Child >16',
                        'c_charge_desc=Viol Injunct Domestic Violence',
                        'c_charge_desc=Viol Injunction Protect Dom Vi',
                        'c_charge_desc=Viol Pretrial Release Dom Viol',
                        'c_charge_desc=Viol Prot Injunc Repeat Viol',
                        'c_charge_desc=Violation License Restrictions',
                        'c_charge_desc=Violation Of Boater Safety Id',
                        'c_charge_desc=Violation of Injunction Order/Stalking/Cyberstalking',
                        'c_charge_desc=Voyeurism',
                        'c_charge_desc=arrest case no charge']]

compas_sensitive_features_dict = {"sex": 0, "race": 2, "age": 1}


def get_compass(sensitive_feature_name, remove_z=False, file_path="/home/btd26/datasets/compas/",
                file_name="compas.npy", **kwargs):
    z_idx = get_z_idx(sensitive_feature_name, compas_sensitive_features_dict)

    # load file
    file_add = os.path.join(file_path, file_name)
    if os.path.exists(file_add):
        M = np.load("/home/btd26/datasets/compas/compas.npy")
    else:
        from aif360.datasets import CompasDataset
        compas = CompasDataset()
        M = np.concatenate([compas.features, compas.labels], axis=1)
        np.save(file_add, M)
    X = M[:, :-1]
    y = M[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z=remove_z)

    return Xtr, Xts, ytr, yts, Ztr, Zts


dataset_fs = [get_german, get_adult, get_bank, get_compass]
dataset_names = ["german", "adult", "bank", "compas"]
n_features_list = [24, 14, 20, 401]
dataset_feature_dict = {"german": german_sensitive_features_dict, "bank": bank_sensitive_features_dict,
                        "adult": adult_sensitive_features_dict, "compas": compas_sensitive_features_dict}

feature_name_dict = {}
for ds in dataset_names:
    keys = dataset_feature_dict[ds]
    feature_name_dict[ds] = {v: k for k, v in dataset_feature_dict[ds].items() for ds in dataset_names}

f_sensitive_list = [sorted(feature_name_dict[ds].keys()) for ds in dataset_names]


def get_n_features_list():
    n_features_list = []
    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        X_test, X_train, Y_test, Y_train = prep_data(Xtr, Xts, ytr, yts, verbose=0)
        n_features = X_train.shape[-1]
        n_features_list.append(n_features)
    return n_features_list

def get_data_list():
    data_list = []
    for i, f in enumerate(dataset_fs):
        Xtr, Xts, ytr, yts, Ztr, Zts = f(0, remove_z=False)
        data_list.append(prep_data(Xtr, Xts, ytr, yts, verbose=0))
    return data_list


def generate_x_labels(dataset_names, f_sensitive_list):
    x_labels = []
    for i, f in enumerate(dataset_names):
        #     print("models #{}".format(len((models_list_p[i]))))
        for j, m in enumerate(f_sensitive_list[i]):
            data_set_name = dataset_names[i]
            feature_name = feature_name_dict[data_set_name][f_sensitive_list[i][j]]
            x_labels.append(data_set_name + "-" + feature_name)
    return x_labels


x_labels = generate_x_labels(dataset_names, f_sensitive_list)


# f_sensitive_list = [[8, 12], [9, 0, 8], [0, 2], [0, 2]]


def prep_data(Xtr, Xts, ytr, yts, verbose=1):
    from tensorflow.keras.utils import to_categorical
    X_train = np.hstack([Xtr])
    Y_train = to_categorical(ytr)
    X_test = np.hstack([Xts])
    Y_test = to_categorical(yts)

    for x in [X_train, X_test, Y_train, Y_test]:
        if verbose > 1:
            print(x.shape)

    return X_test, X_train, Y_test, Y_train


def extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z):
    if remove_z:
        ix = np.delete(np.arange(X_train.shape[1]), z_idx)
    else:
        ix = np.arange(X_train.shape[1])
    Xtr = X_train[:, ix]
    Ztr = X_train[:, z_idx].reshape(-1, 1)
    Xts = X_test[:, ix]
    Zts = X_test[:, z_idx].reshape(-1, 1)
    ytr = y_train
    yts = y_test
    return Xtr, Xts, Ztr, Zts, ytr, yts


def nulify_feature(X_train, Y_train, i):
    x = np.copy(X_train[:, :])
    x[:, i] = 0
    return x, Y_train


binarise_dict = {"german": {
    "gender": {"targets": [[2], [1, 3, 4]], "assign_vals": [0, 1]},
    "age": {"targets": [[1], [2]], "assign_vals": [0, 1]},
},
    "adult":
        {"gender": {"targets": None, "assign_vals": [0, 1]},
         "age": {"targets": lambda x: x > -0.99570562, "assign_vals": [0, 1]},
         "race": {"targets": lambda x: x > 0.39, "assign_vals": [0, 1]}, }
    ,
    "bank": {"age": {"targets": lambda x: x > -1.44169297e+00, "assign_vals": [0, 1]},
             "marital": {"targets": lambda x: x == np.unique(x)[2], "assign_vals": [0, 1]}, },
    "compas": {"age": {"targets": lambda x: x > 25, "assign_vals": [0, 1]},
               "sex": {"targets": [[0], [1]], "assign_vals": [1, 0]},
               "race": {"targets": [[0], [1]], "assign_vals": [0, 1]},
               }
}


def sample_data(data, sample, use_train):
    X_test, X_train, Y_test, Y_train = data
    if use_train:
        X = X_train
        Y = Y_train

    else:
        X = X_test
        Y = Y_test
    # SAMPLE
    if sample is not None:
        random.seed(30)
        n_samples = int(X.shape[0])
        ix_sample = random.sample(range(n_samples), min(sample, n_samples))
        X = X[ix_sample, :]
        Y = Y[ix_sample, :]
    return X, Y