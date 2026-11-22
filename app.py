"""SentimentSense Pro — Streamlit + pretrained DistilBERT + SQLite.
No model training is claimed. Demo dataset is synthetic and small.
"""
import io
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'reviews.csv'
DB = ROOT / 'sentimentsense.db'
MODEL_ID = 'distilbert-base-uncased-finetuned-sst-2-english'
COLORS = {'Positive':'#27d9a0', 'Negative':'#ff6584', 'Neutral':'#f8bf64'}
STOP = set('the a an and or is are was were to for with it this that of in on my i you we they very but have has had at as from by so be not its our their your'.split())

st.set_page_config(page_title='SentimentSense Pro | AI Intelligence Studio', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 75% 0%,rgba(69,86,157,.19),transparent 34%),#0b1020;color:#edf2ff}
section[data-testid="stSidebar"]{background:#10182c;border-right:1px solid #27344c}
section[data-testid="stSidebar"] *{color:#e4ecff}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em}
h1{font-size:2.55rem!important}h2{font-size:1.55rem!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#18243d,#121b30);border:1px solid #2b3c5a;padding:19px;border-radius:18px}
[data-testid="stMetricLabel"]{color:#a9b8d1}
[data-testid="stMetricValue"]{color:#f5f8ff}
div.stButton>button[kind="primary"],div.stDownloadButton>button[kind="primary"]{background:linear-gradient(100deg,#6674ff,#a55cff);border:0;border-radius:12px;color:white;font-weight:700;padding:.7rem 1.3rem}
div.stButton>button{border-radius:12px}
[data-testid="stFileUploader"], [data-testid="stDataFrame"], .stTextArea textarea{border-radius:15px}
.hero{padding:28px 32px;border:1px solid #354369;border-radius:24px;background:linear-gradient(110deg,#1d2b51,#172044 58%,#38215b);margin-bottom:22px}
.eyebrow{color:#9dcbff;font-weight:700;letter-spacing:.14em;font-size:.72rem;text-transform:uppercase}
.hero h1{margin:.35rem 0 .4rem;font-size:2.55rem;color:#fff}.hero p{color:#becce6;font-size:1.04rem;margin:0}
.chip{display:inline-block;border:1px solid #5868a2;color:#cbd6ff;border-radius:999px;padding:5px 11px;margin:14px 7px 0 0;font-size:.77rem}
.note{padding:13px 16px;background:#17233b;border:1px solid #31425e;border-radius:12px;color:#c4d0e6;font-size:.88rem;margin:10px 0 18px}
[data-testid="stTabs"] button{font-weight:600}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading DistilBERT (first run downloads ~250 MB)...')
def load_model():
    from transformers import pipeline
    return pipeline('text-classification', model=MODEL_ID, framework='pt', device=-1)

def db_conn():
    conn = sqlite3.connect(DB, timeout=15)
    conn.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, review TEXT NOT NULL, sentiment TEXT NOT NULL, confidence REAL NOT NULL, source TEXT NOT NULL)''')
    conn.commit()
    return conn

def save_results(rows, source):
    with db_conn() as conn:
        conn.executemany('INSERT INTO predictions (created_at,review,sentiment,confidence,source) VALUES (?,?,?,?,?)', [(datetime.now().isoformat(timespec='seconds'),r['review'],r['sentiment'],r['confidence'],source) for r in rows])

def history():
    with db_conn() as conn:
        return pd.read_sql_query('SELECT * FROM predictions ORDER BY id DESC',conn)

def predict(texts, batch_size=12):
    model=load_model()
    out=[]
    for start in range(0,len(texts),batch_size):
        chunk=texts[start:start+batch_size]
        results=model(chunk, truncation=True, max_length=512, batch_size=min(batch_size,len(chunk)))
        for review,result in zip(chunk,results):
            out.append({'review':review,'sentiment':str(result['label']).title(),'confidence':float(result['score'])})
    return out

def safe_predict(texts):
    try:
        return predict(texts)
    except Exception as exc:
        st.error(f'Could not load/run the AI model: {exc}')
        st.info('Check internet for the first model download. If memory is low, try a smaller batch. No fake predictions are substituted.')
        return None

def load_data():
    try:
        df=pd.read_csv(DATA)
        if not {'review','sentiment'}.issubset(df.columns):
            st.error('Bundled dataset must contain review and sentiment columns.')
            return pd.DataFrame(columns=['review','sentiment'])
        df=df[['review','sentiment']].dropna().copy()
        df['review']=df['review'].astype(str)
        df['sentiment']=df['sentiment'].astype(str).str.strip().str.title()
        return df[df['review'].str.strip().ne('')].reset_index(drop=True)
    except Exception as exc:
        st.error(f'Cannot read bundled dataset: {exc}')
        return pd.DataFrame(columns=['review','sentiment'])

def chart(fig, key=None):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#d5def2'),margin=dict(l=12,r=12,t=50,b=12),legend_title_text='')
    st.plotly_chart(fig,use_container_width=True,key=key)

def distribution(df):
    if df.empty: st.info('No reviews available.'); return
    counts=df['sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Reviews')
    c1,c2=st.columns(2)
    with c1:
        chart(px.bar(counts,x='Sentiment',y='Reviews',color='Sentiment',color_discrete_map=COLORS,title='Sentiment breakdown',text='Reviews'))
    with c2:
        chart(px.pie(counts,names='Sentiment',values='Reviews',hole=.63,color='Sentiment',color_discrete_map=COLORS,title='Sentiment share'))

def words(series, n=12):
    tokens=[]
    for s in series.fillna('').astype(str):
        tokens.extend(w for w in re.findall(r"[a-z']+",s.lower()) if len(w)>2 and w not in STOP)
    return Counter(tokens).most_common(n)

def cloud(series):
    from wordcloud import WordCloud
    txt=' '.join(series.fillna('').astype(str))
    if not txt.strip(): st.info('No words available.');return
    wc=WordCloud(width=1100,height=440,background_color='#121b30',colormap='cool',stopwords=STOP).generate(txt)
    st.image(wc.to_array(),use_container_width=True)

def heading(title,subtitle=''):
    st.subheader(title)
    if subtitle: st.caption(subtitle)

st.sidebar.markdown('## 🧠 SentimentSense **PRO**')
st.sidebar.caption('TRANSFORMER INTELLIGENCE STUDIO')
st.sidebar.divider()
PAGES=['🏠 Command Center','⚡ Live AI Lab','📁 Batch Analyzer','📊 Dataset Explorer','🎯 Model Evaluation','☁️ Text Intelligence','🕒 Prediction Vault','📚 Project & Viva']
page=st.sidebar.radio('WORKSPACE',PAGES,label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.success('● DistilBERT • Binary Sentiment')
st.sidebar.caption('Model: SST-2 fine-tuned DistilBERT')
st.sidebar.caption('Storage: local SQLite | No account required')
st.sidebar.caption('Tip: first prediction downloads model.')

df=load_data()
if page==PAGES[0]:
    st.markdown('''<div class="hero"><div class="eyebrow">AI-POWERED CUSTOMER INTELLIGENCE</div><h1>Understand every review.<br>Discover the bigger picture.</h1><p>Explore sentiment trends, analyze reviews live, and turn feedback into insights using DistilBERT.</p><span class="chip">⚡ Live predictions</span><span class="chip">📈 Interactive analytics</span><span class="chip">🔒 Local database</span></div>''',unsafe_allow_html=True)
    h=history()
    a,b,c,d=st.columns(4)
    a.metric('Dataset reviews',f'{len(df):,}')
    b.metric('Positive share',f'{(df.sentiment.eq("Positive").mean()*100):.1f}%' if len(df) else '—')
    c.metric('Negative reviews',f'{df.sentiment.eq("Negative").sum():,}')
    d.metric('Saved AI predictions',f'{len(h):,}')
    heading('Sentiment landscape','Interactive charts based on bundled labeled demo reviews — not model predictions.')
    distribution(df)
    if not df.empty:
        a,b=st.columns([1,1])
        with a:
            df2=df.copy();df2['Words']=df2.review.str.split().str.len()
            chart(px.histogram(df2,x='Words',color='sentiment',color_discrete_map=COLORS,nbins=15,title='Review length distribution'))
        with b:
            common=pd.DataFrame(words(df.review),columns=['Word','Frequency'])
            if not common.empty: chart(px.bar(common,x='Frequency',y='Word',orientation='h',title='Most common words'))
    st.markdown('<div class="note">ℹ️ Dataset is a small synthetic demonstration set. These charts describe its labels, not real customer trends or model accuracy.</div>',unsafe_allow_html=True)

elif page==PAGES[1]:
    heading('⚡ Live AI Lab','Paste any English customer review and get a DistilBERT sentiment prediction.')
    samples=['The quality is fantastic and delivery was super fast!','Terrible experience, the product stopped working after one day.','I love the camera but the battery is disappointing.']
    choice=st.selectbox('Try a sample', ['Write my own']+samples)
    review=st.text_area('Customer review',value='' if choice=='Write my own' else choice,height=150,placeholder='Type a product review here...',max_chars=6000)
    c1,c2=st.columns([1,2]);c1.caption(f'{len(review)} / 6000 characters')
    if st.button('✨ Analyze with DistilBERT',type='primary',use_container_width=True):
        if not review.strip():st.warning('Enter a review first.')
        else:
            result=safe_predict([review.strip()])
            if result:
                st.session_state['live_result']=result[0]
                save_results(result,'Live Lab')
    r=st.session_state.get('live_result')
    if r:
        st.divider()
        c1,c2,c3=st.columns(3)
        c1.metric('Prediction',r['sentiment'])
        c2.metric('Model score',f'{r["confidence"]:.1%}')
        c3.metric('Review words',len(r['review'].split()))
        st.progress(min(1.,r['confidence']))
        st.info('The score is a model confidence output, not calibrated accuracy. Mixed or neutral reviews are still forced into Positive/Negative by this binary model.')
        st.code(r['review'],language=None)

elif page==PAGES[2]:
    heading('📁 Batch Analyzer','Upload your own CSV and classify many reviews in one click.')
    st.markdown('<div class="note">Upload a UTF-8 CSV with a review/text column. For a fast laptop demo, start with 10–30 rows. Maximum 200 reviews per run.</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader('Choose CSV file',type=['csv'])
    source=uploaded if uploaded else DATA
    try: batch=pd.read_csv(source)
    except Exception as exc:st.error(f'Invalid CSV: {exc}');batch=pd.DataFrame()
    if not batch.empty:
        column=st.selectbox('Which column contains the reviews?',list(batch.columns),index=list(batch.columns).index('review') if 'review' in batch.columns else 0)
        limit=st.slider('Rows to analyze',1,min(len(batch),200),min(len(batch),30))
        st.dataframe(batch.head(8),use_container_width=True,hide_index=True)
        if st.button('🚀 Analyze batch',type='primary'):
            texts=batch[column].fillna('').astype(str).head(limit).str.strip()
            texts=texts[texts.ne('')].tolist()
            if not texts:st.warning('No non-empty reviews found.')
            else:
                with st.spinner(f'Classifying {len(texts)} reviews...'):
                    out=safe_predict(texts)
                if out:
                    save_results(out,'Batch CSV')
                    st.session_state['batch_output']=pd.DataFrame(out)
        result=st.session_state.get('batch_output')
        if result is not None:
            st.success(f'Analyzed {len(result)} reviews. Predictions saved to SQLite.')
            distribution(result)
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.download_button('⬇️ Download classified CSV',result.to_csv(index=False).encode('utf-8'),'classified_reviews.csv','text/csv',type='primary')

elif page==PAGES[3]:
    heading('📊 Dataset Explorer','Explore, search, filter and export the included labeled CSV.')
    if not df.empty:
        a,b=st.columns([1,2]);labels=a.multiselect('Sentiment labels',sorted(df.sentiment.unique()),default=sorted(df.sentiment.unique()));query=b.text_input('Search review text',placeholder='Try: delivery, product, quality...')
        filtered=df[df.sentiment.isin(labels)]
        if query:filtered=filtered[filtered.review.str.contains(query,case=False,regex=False,na=False)]
        st.metric('Matching reviews',len(filtered))
        st.dataframe(filtered,use_container_width=True,hide_index=True,height=360)
        st.download_button('⬇️ Export filtered CSV',filtered.to_csv(index=False).encode('utf-8'),'filtered_reviews.csv','text/csv')
        distribution(filtered)

elif page==PAGES[4]:
    heading('🎯 Model Evaluation','Measure real predictions against known labels. No hard-coded scores.')
    st.markdown('<div class="note">Use a separate labeled evaluation CSV for meaningful metrics. Bundled reviews are synthetic, tiny and only useful for a functionality check — not a valid performance benchmark.</div>',unsafe_allow_html=True)
    file=st.file_uploader('Optional labeled test CSV (review,sentiment)',type=['csv'],key='evaluation')
    try: evaluation=pd.read_csv(file) if file else df.copy()
    except Exception as exc:st.error(str(exc));evaluation=pd.DataFrame()
    if {'review','sentiment'}.issubset(evaluation.columns):
        evaluation=evaluation.dropna(subset=['review','sentiment']).copy()
        evaluation['sentiment']=evaluation.sentiment.astype(str).str.strip().str.title()
        evaluation=evaluation[evaluation.sentiment.isin(['Positive','Negative']) & evaluation.review.astype(str).str.strip().ne('')]
        if len(evaluation):
            count=st.slider('Evaluate number of rows',1,min(200,len(evaluation)),min(30,len(evaluation)))
            if st.button('🧪 Run model evaluation',type='primary'):
                with st.spinner('Running labeled evaluation...'):
                    preds=safe_predict(evaluation.review.astype(str).head(count).tolist())
                if preds:
                    from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
                    truth=evaluation.sentiment.head(count).tolist(); pred=[x['sentiment'] for x in preds]
                    precision,recall,f1,_=precision_recall_fscore_support(truth,pred,average='binary',pos_label='Positive',zero_division=0)
                    st.session_state['eval_result']={'truth':truth,'pred':pred,'accuracy':accuracy_score(truth,pred),'precision':precision,'recall':recall,'f1':f1,'reviews':evaluation.review.head(count).tolist(),'uploaded':file is not None}
        else:st.warning('No valid labeled rows. Use Positive/Negative labels.')
    else:st.warning('Evaluation CSV must have review and sentiment columns.')
    result=st.session_state.get('eval_result')
    if result:
        st.caption('Source: '+('uploaded labeled CSV' if result['uploaded'] else 'bundled synthetic demo CSV')+' • '+str(len(result['truth']))+' evaluated rows')
        a,b,c,d=st.columns(4)
        for col,name,key in [(a,'Accuracy','accuracy'),(b,'Precision (+)','precision'),(c,'Recall (+)','recall'),(d,'F1 (+)','f1')]:col.metric(name,f'{result[key]:.1%}')
        from sklearn.metrics import confusion_matrix
        labels=['Negative','Positive']; cm=confusion_matrix(result['truth'],result['pred'],labels=labels)
        fig=px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale='Blues',labels={'x':'Predicted','y':'Actual','color':'Count'},title='Confusion matrix (actual vs predicted)')
        chart(fig)
        report=pd.DataFrame({'review':result['reviews'],'actual':result['truth'],'predicted':result['pred']});report['correct']=report.actual.eq(report.predicted)
        st.dataframe(report,use_container_width=True,hide_index=True)
        st.download_button('⬇️ Export evaluation results',report.to_csv(index=False),'evaluation_results.csv','text/csv')

elif page==PAGES[5]:
    heading('☁️ Text Intelligence','Discover common words and compare language across sentiment groups.')
    if not df.empty:
        chosen=st.selectbox('Analyze reviews', ['All']+sorted(df.sentiment.unique()))
        filtered=df if chosen=='All' else df[df.sentiment.eq(chosen)]
        cloud(filtered.review)
        common=pd.DataFrame(words(filtered.review,20),columns=['Word','Frequency'])
        if not common.empty:chart(px.bar(common,x='Frequency',y='Word',orientation='h',title='Top words (frequency)'))
        st.markdown('#### Keyword explorer')
        term=st.text_input('Find reviews mentioning a keyword',placeholder='battery')
        if term:st.dataframe(filtered[filtered.review.str.contains(term,case=False,regex=False,na=False)],use_container_width=True,hide_index=True)
        st.caption('Word frequencies are descriptive; they are not Transformer explanations or feature importance.')

elif page==PAGES[6]:
    heading('🕒 Prediction Vault','Your predictions are stored locally in SQLite and remain available after restart.')
    h=history()
    if h.empty:st.info('No predictions saved yet. Use Live AI Lab or Batch Analyzer first.')
    else:
        a,b,c=st.columns(3);a.metric('Saved predictions',len(h));b.metric('Positive predictions',h.sentiment.eq('Positive').sum());c.metric('Average model score',f'{h.confidence.mean():.1%}')
        labels=st.multiselect('Filter by sentiment',sorted(h.sentiment.unique()),default=sorted(h.sentiment.unique()))
        filtered=h[h.sentiment.isin(labels)]
        st.dataframe(filtered,use_container_width=True,hide_index=True,height=350)
        st.download_button('⬇️ Export prediction history',filtered.to_csv(index=False).encode('utf-8'),'prediction_history.csv','text/csv')
        distribution(filtered)
        with st.expander('⚠️ Clear all local prediction history'):
            st.warning('This permanently deletes saved predictions from the local SQLite database.')
            if st.button('Delete all history'):
                with db_conn() as conn:conn.execute('DELETE FROM predictions')
                st.rerun()

else:
    heading('📚 Project Guide & Viva','Everything you need to explain your project in college.')
    st.markdown('''**Project title:** SentimentSense Pro — Transformer-Based Customer Review Sentiment Analytics

**Problem:** Manually reading many reviews takes time. This app uses a pretrained Transformer to classify English text as Positive or Negative and provides a dashboard for exploration.

**Workflow:** Review → tokenizer → DistilBERT (SST-2) → predicted label + score → SQLite → dashboard.

**Tech stack:** Python, Streamlit, Hugging Face Transformers, PyTorch, Pandas, Plotly, scikit-learn, WordCloud, SQLite.

**What is DistilBERT?** A smaller, faster distilled version of BERT. This checkpoint is fine-tuned for two-class sentiment classification on SST-2.

**What is tokenization?** Splitting text into model-readable token IDs, including subword tokens.

**What is the database for?** SQLite stores predictions, model scores and timestamps for later inspection.

**What does the accuracy mean?** The fraction of correctly predicted labeled test examples. It is only meaningful for the dataset actually evaluated.

**What is a confusion matrix?** A 2×2 table of actual vs predicted classes, revealing types of classification errors.

**Is the model trained in this project?** No. This project runs an existing pretrained/fine-tuned model. The evaluation page measures it on labeled examples.

**Limitations:** English-focused; binary labels only; neutral/mixed sentiment may be misclassified; 512-token input limit; first model download needs internet; synthetic bundled dataset is not representative of real-world performance.

**Future scope:** Multilingual models, three-class sentiment, aspect-based sentiment, real labeled evaluation sets, and domain-specific fine-tuning.''')
    st.code('python -m streamlit run app.py',language='powershell')
