#!/usr/bin/env python3
import os, re, argparse, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
PROJECTS = ['TensorFlow','PyTorch','Keras','MXNet','Caffe']
N_RUNS, TEST_SIZE, RANDOM_SEED = 30, 0.30, 42

def clean_text(text):
    if not isinstance(text, str): return ''
    for pat, rep in [(r'https?://\S+', ' '),(r'<[^>]+>', ' '),(r'[^a-zA-Z\s]', ' ')]:
        text = re.sub(pat, rep, text)
    return ' '.join(text.lower().split())

def load_project(data_dir, project):
    df = pd.read_csv(os.path.join(data_dir, f'{project}.csv'))
    tc = next((c for c in df.columns if c.lower() in ['title_body','text','body','title']), df.columns[0])
    lc = next((c for c in df.columns if c.lower() in ['label','class','target']), df.columns[-1])
    return df[tc].apply(clean_text).values, df[lc].fillna(0).astype(int).values

def run_project(project, data_dir):
    texts, labels = load_project(data_dir, project)
    raw = {m:{k:[] for k in ['precision','recall','f1']} for m in ['baseline','solution']}
    for seed in range(N_RUNS):
        Xtr,Xte,ytr,yte = train_test_split(texts,labels,test_size=TEST_SIZE,random_state=seed,stratify=labels)
        for tag, vec, clf in [
            ('baseline', TfidfVectorizer(stop_words='english',max_features=5000), MultinomialNB()),
            ('solution', TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words='english',max_features=15000),
             RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=RANDOM_SEED,n_jobs=-1))]:
            clf.fit(vec.fit_transform(Xtr), ytr)
            yp = clf.predict(vec.transform(Xte))
            raw[tag]['precision'].append(precision_score(yte,yp,zero_division=0))
            raw[tag]['recall'].append(recall_score(yte,yp,zero_division=0))
            raw[tag]['f1'].append(f1_score(yte,yp,zero_division=0))
    return raw

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_dir', default='results')
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    for proj in PROJECTS:
        raw = run_project(proj, a.data_dir)
        for app in ['baseline','solution']:
            pd.DataFrame(raw[app]).to_csv(os.path.join(a.output_dir, f'{proj}_{app}_raw.csv'), index=False)
        for m in ['precision','recall','f1']:
            s,b = raw['solution'][m], raw['baseline'][m]
            try: pv = stats.wilcoxon(s,b,alternative='two-sided')[1]
            except: pv = 1.0
            a12 = sum(1 if x>y else 0.5 if x==y else 0 for x in s for y in b)/(len(s)*len(b))
            print(f'{proj} {m}: B={np.mean(b):.3f} S={np.mean(s):.3f} p={pv:.4f} A12={a12:.3f}')

if __name__ == '__main__': main()
