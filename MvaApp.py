import io
import streamlit as st
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import altair as alt
import pydeck as pdk
from streamlit import caching
import streamlit.components.v1 as components
from statsmodels.iolib.summary2 import summary_col

st.set_option('deprecation.showfileUploaderEncoding', False)
st.beta_set_page_config(
    page_title="MVA App",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
 )
st.markdown('<style>h1{color: darkmagenta;}</style>', unsafe_allow_html=True)
st.markdown('<style>h2{color: green;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3{color: crimson;}</style>', unsafe_allow_html=True)
st.markdown('<style>p{color: darkmagenta;}</style>', unsafe_allow_html=True)


st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')
st.write("""
# Calculating EUR using Multivariate Regression Model and It's Statistical Analysis """+'<i class="material-icons">insights</i>', unsafe_allow_html=True)
st.write("""
## Snapshot of Pre-Loaded/Uploaded Data:"""+'<i class="material-icons">preview</i>', unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Choose a CSV file (Optional)", type="csv")

if uploaded_file is not None:
   text_io = io.TextIOWrapper(uploaded_file)
   data = pd.read_csv(uploaded_file, encoding= 'unicode_escape')
   st.write(data.head())
else:  
   data = pd.read_csv("BemisData.csv", encoding= 'unicode_escape')
   st.write(data.head())
st.write('To upload your own data refer to user input section (Columns should be same)')
st.markdown("""## Let us visualize our Data: """+'<i class="material-icons">map</i>', unsafe_allow_html=True)
a=2
@st.cache(suppress_st_warning=False, max_entries=200)
def proceed(): 
  if a==1:
    return True
  else:
    return False
if st.button('Proceed'):
 caching.clear_cache()
 a=1
 proceed() 

if proceed():
 maplayer = st.sidebar.radio(
     "Select the Type of Map?",
     ('ColumnLayer', 'HeatmapLayer'))
 
 genre = st.sidebar.selectbox(
     "Which property to Map?",
     ('Pressure', 'K', 'So','Sw', 'Porosity','Skin', 'Shaliness'))
 
 Map = pd.DataFrame(data, columns=['KID','LONGITUDE','LATITUDE',genre])
 
 maxx=Map[genre].max()
 maxx=maxx/1000
 
 st.write("""
 ### MAP View of: """ + genre)
 agree = st.checkbox('Do not want Dark Mode')
 style='mapbox://styles/rahultiwari989/ckdzx6a1u0vfe19o76mxa1pss'
 if agree:
    style='mapbox://styles/mapbox/light-v9'
 else:
    style='mapbox://styles/rahultiwari989/ckdzx6a1u0vfe19o76mxa1pss'
 
 st.pydeck_chart(pdk.Deck(
     map_style=style,
     initial_view_state=pdk.ViewState(
         latitude=39.08,
         longitude=-99.22,
         zoom=11,
         pitch=50,
    ),
     layers=[
         pdk.Layer(
          maplayer,
          data=Map,
          get_position='[LONGITUDE,LATITUDE]',
          diskResolution= 360,
          radius= 250,
          elevationScale= 4,
          opacity=5,
          stroked=False,
          filled=True,
          extruded=True,
          wireframe=True,
          get_elevation=genre +'/'+str(maxx) ,
          get_fill_color='['+genre +'/'+str(maxx)+'/100%5*225,'+genre +'/'+str(maxx)+'/100%4*225, '+genre +'/'+str(maxx)+'/100%3*225]',
          get_line_color='['+genre +'/'+str(maxx)+'/100%5*225,'+genre +'/'+str(maxx)+'/100%4*225, '+genre +'/'+str(maxx)+'/100%3*225]',
          pickable=True,
          aggregation='"MEAN"',
          get_weight=genre +'/'+str(maxx),
    
          autoHighlight=True,
    
          intensity= 2,
    
          radiusPixels= 25,
          threshold= 0.05,
          visible= True,
          wrapLongitude=True),
        ],
    tooltip = {
   "html": "<b>KID: </b> {KID}"
            "<br/> <b>Latitude: </b> {LATITUDE}"
            " <br/> <b>Longitude: </b> {LONGITUDE} "
            "<br/> <b>"+genre+" value: </b> {"+genre+"}",
   "style": {
        "backgroundColor": "darkmagenta",
        "color": "white"
    }
  }
  ))
else:
   st.stop()

st.markdown("""## Now we will determine the no. of clusters required for appropriate representation of the data: """+'<i class="material-icons">bubble_chart</i>', unsafe_allow_html=True)
a=2
@st.cache()
def plot(): 
  if a==1:
    return True
  else:
    return False
if st.button('Plot SSE vs No. of Clusters'):
 caching.clear_cache()
 a=1
 proceed()
 plot() 
features = pd.DataFrame(data, columns=['Pressure', 'K', 'So','Sw', 'Porosity','Skin', 'Shaliness'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

if plot():
 kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": None,
 }

 # A list holds the SSE values for each k
 sse = []
 for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

 source = pd.DataFrame(sse, columns=['SSE'], index=pd.RangeIndex(1,11, name='x'))
 source = source.reset_index().melt('x', var_name='category', value_name='y')

 line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
    alt.Y('y', title='SSE'),
    alt.X('x', title='Number of Clusters'),
    color='category:N',
    ).properties(
    title='SSE vs No. of Clusters',
    
 )
 
 st.altair_chart(line_chart, use_container_width=True)
 

else:
   st.stop()

a=2
@st.cache(suppress_st_warning=False)
def knee(): 
  if a==1:
    return True
  else:
    return False
if st.button('Knee Locator for above graph'):
 caching.clear_cache()
 a=1
 proceed()
 plot()
 knee() 

if knee():
 kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")

 clusters = kl.elbow

 st.markdown("""### Automatically generated optimum no. of Clusters:    """+str(clusters))
 st.write('To change the no. of clusters manually refer to user input section')
 clusters=st.sidebar.slider('Clusters (can be changed here manually)', 1, 10, int(clusters))
 

else:
   st.stop()

st.markdown("""## Let's perform K-means clustering by clicking below: """+'<i class="material-icons">blur_on</i>', unsafe_allow_html=True)

a=2
@st.cache(suppress_st_warning=True)
def kmean(): 
  if a==1:
    return True
  else:
    return False
if st.button('Kmeans'):
 caching.clear_cache()
 a=1
 proceed()
 plot()
 knee() 
 kmean() 

if kmean():
 kmeans = KMeans(
    init="k-means++",
    n_clusters= clusters,
    n_init=10,
    max_iter=300,
    random_state=None
  )
 kmeans.fit(scaled_features)
 features['Cluster']=kmeans.labels_+1
 features['B']= data.B
 features['D']= data.D
 features['IP(DCA)']= data['IP(DCA)']
 features['LONGITUDE']=data['LONGITUDE']
 features['LATITUDE']=data['LATITUDE']
 features['KID']=data['KID']
 st.markdown("""### Here are the results: """)
 st.write(features['Cluster'].value_counts())
else:
  st.stop()
st.markdown("""## Click below to map the clustering results: """+'<i class="material-icons">map</i>', unsafe_allow_html=True)
a=2
@st.cache(suppress_st_warning=True)
def mapkmeans(): 
  if a==1:
    return True
  else:
    return False
if st.button('MAP Kmeans'):
 caching.clear_cache()
 a=1
 proceed()
 plot()
 knee() 
 kmean()
 mapkmeans() 

if mapkmeans():
 Map2 = pd.DataFrame(features, columns=['KID','LONGITUDE','LATITUDE','Cluster'])
 
 st.write("""
 ### MAP View: Cluster""")
 agree = st.checkbox('I do not want Dark Mode')
 style='mapbox://styles/rahultiwari989/ckdzx6a1u0vfe19o76mxa1pss'
 if agree:
    style='mapbox://styles/mapbox/light-v9'
 else:
    style='mapbox://styles/rahultiwari989/ckdzx6a1u0vfe19o76mxa1pss'
 
 st.pydeck_chart(pdk.Deck(
     map_style=style,
     initial_view_state=pdk.ViewState(
         latitude=39.08,
         longitude=-99.22,
         zoom=11,
         pitch=50,
    ),
     layers=[
         pdk.Layer(
         'ColumnLayer',
          Map2,
          get_position='[LONGITUDE,LATITUDE]',
          diskResolution= 360,
          radius= 250,
          elevationScale= 4,
          opacity=5,
          stroked=False,
          filled=True,
          extruded=True,
          wireframe=True,
          get_elevation='(Cluster)*180',
          
          get_fill_color='[Cluster%4*225,Cluster%3*225,Cluster%2*225]',
          get_line_color='[Cluster%4*225,Cluster%3*225,Cluster%2*225]',
          pickable=True,
          
    
          autoHighlight=True,
    
          intensity= 2,
    
          radiusPixels= 25,
          threshold= 0.05,
          visible= True,
          wrapLongitude=True
         ),
         
     ],
     tooltip = {
   "html": "<b>KID: </b> {KID}"
            "<br/> <b>Latitude: </b> {LATITUDE}"
            " <br/> <b>Longitude: </b> {LONGITUDE} "
            "<br/> <b>Cluster assigned: </b> {Cluster}",
   "style": {
        "backgroundColor": "darkmagenta",
        "color": "white"
    }
  }
  )) 
else:
 st.stop() 
st.markdown("""## Let's create Multivariate Regression Model: """+'<i class="material-icons">mediation</i>', unsafe_allow_html=True)
a=2
@st.cache(suppress_st_warning=True)
def mvr(): 
  if a==1:
    return True
  else:
    return False
if st.button('Perform MVR'):
 caching.clear_cache()
 a=1
 proceed()
 plot()
 knee() 
 kmean()
 mapkmeans() 
 mvr() 

if mvr():
 def eur(IP, B, D):
  EUR = 365*((IP**B)/(D*(1-B)))*(IP**(1-B)-5**(1-B))
  return EUR
 results=pd.DataFrame(columns=['Original EUR','Predicted EUR','Error'])
 model = []
 with open('summary.txt', 'w') as fh: 
  fh.write('\t WELCOME!! \n \n')
 u=0
 st.write("""### Showing Results for all Clusters:  """)
 for i in range(1,clusters+1):
  l=features['Cluster'].value_counts()[i]
  k=int(l*0.2)
  df = features.loc[features['Cluster'] == i]
  tf = features.loc[features['Cluster'] == i][0:k]
  df['EUR']= eur(IP=df['IP(DCA)'], B=df['B'], D= df['D'])
  t=df['EUR'].sum()
  X = df[['Pressure', 'K', 'So','Sw', 'Porosity','Skin', 'Shaliness']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
  Y = df['B']
  # with sklearn
  regr = linear_model.LinearRegression()
  regr.fit(X, Y)

  # prediction with sklearn
  A= tf[['Pressure', 'K', 'So','Sw', 'Porosity','Skin', 'Shaliness']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
  tf['Pred_B']=regr.predict(A)
  # print ('Predicted B: \n', regr.predict(A))


  Y = df['D']
  # with sklearn
  regr = linear_model.LinearRegression()
  regr.fit(X, Y)
  tf['Pred_D']=regr.predict(A)
  #print ('Predicted D: \n', regr.predict(A))

  Y = df['IP(DCA)']
  # with sklearn
  regr = linear_model.LinearRegression()
  regr.fit(X, Y)
  tf['Pred_IP']=regr.predict(A)
  #print ('Predicted IP: \n', regr.predict(A))
  tf['EUR']= eur(IP=tf['IP(DCA)'], B=tf['B'], D= tf['D'])

  s=tf['EUR'].sum()

  tf['Pred_EUR']= eur(IP=tf['Pred_IP'], B=tf['Pred_B'], D= tf['Pred_D'])
  p=tf['Pred_EUR'].sum()

  results = results.append({'Original EUR':t,'Predicted EUR':(t-s+p),'Error':((p-s)/t)}, ignore_index=True)
   # with statsmodels
  X = sm.add_constant(X) # adding a constant

  Y = df['B']
  model.append(sm.OLS(Y, X).fit())
  predictions1 = model[u].predict(X) 
  u=u+1
  Y = df['D']
  model.append(sm.OLS(Y, X).fit())
  predictions2 = model[u].predict(X) 
  u=u+1
  Y = df['IP(DCA)']
  model.append(sm.OLS(Y, X).fit())
  predictions3 = model[u].predict(X) 
  u=u+1
  with open('summary.txt', 'a') as fh: 
   fh.write('Result Summary for Cluster '+str(i)+' \n \n')
   fh.write(model[u-3].summary().as_text())
   fh.write('\n \n')
   fh.write(model[u-2].summary().as_text())
   fh.write('\n \n')
   fh.write(model[u-1].summary().as_text()) 
   fh.write('\n \n')
   fh.close()

   
 ppp= summary_col(results=model, float_format='%.4f', stars=False,
                info_dict=None, drop_omitted=False)
 ppp.add_title('Table 1 - OLS Regressions')
 st.write(ppp.tables[0])
 st.write('Std. errors have been indicated in parenthesis')
 data1 = open('summary.txt', 'rb').read()
 b64 = base64.b64encode(data1).decode('UTF-8')
 #b64 = base64.b64encode(data.as_text().encode()).decode()  # some strings <-> bytes conversions necessary here
 href = f'<a href="data:file/txt;base64,{b64}">Click Here To Download Complete MVR Analysis Summary!</a> (right-click and save as &lt;some_name&gt;.txt)'
 st.markdown(href, unsafe_allow_html=True)
 components.html("""<HTML>
 <BODY>
<form method="get" action="https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv">
   <button type="submit">Click Here To Download Complete MVR Analysis Summary!</button>
</form>
</BODY>
</HTML>
""",height=30,width=500)   
else:
 st.stop()

st.markdown('<style>body{color: black; font-weight:bold;}</style>', unsafe_allow_html=True)
st.markdown('<style>td{color: darkmagenta;}</style>', unsafe_allow_html=True)
st.markdown("""### Click below to generate Results""")
a=2
@st.cache(suppress_st_warning=True)
def result(): 
  if a==1:
    return True
  else:
    return False
if st.button("""Display Final Results"""):
 caching.clear_cache()
 a=1
 proceed()
 plot()
 knee() 
 kmean()
 mapkmeans() 
 mvr() 
 result() 

if result():
 st.write(results)
 st.success('Results were successfully generated!!')
 st.balloons()
else:
  st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown('Made with '+'<span class="material-icons">favorite</span>'+' by Rahul Tiwari', unsafe_allow_html=True)
