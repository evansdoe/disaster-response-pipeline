import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib 
import joblib
from sqlalchemy import create_engine
from custom_transformer import DisasterWordExtrator, replace_urls, tokenize


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_categories', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    res_df = df.drop(columns=['id', 'message', 'original', 'genre']);
    
    res_count = res_df.sum().sort_values(ascending=False)[:10]
    
    res_categories = res_count.index

    coln_labels = df.drop(columns=['id', 'message', 'original', 'genre']).sum().sort_values(ascending=False).index

    df_genre_grps = df.groupby('genre')[coln_labels].sum().reset_index()


    df_genre_grps = df_genre_grps.drop(columns=['genre']).rename(index={0 : 'direct', 1:'news', 2:'social'})


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }, 
                'template': "plotly_dark"
            }
        }, 
        
        {
            'data': [
                Bar(
                    x=res_categories,
                    y=res_count
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Categories"
                }, 
                'template': 'plotly_dark'
            }
        },
        
        {
    		'data': [
    					{
    						'name': 'direct',
              				'type': 'bar',
              				'x': coln_labels[:10],
              				'xaxis': 'x',
              				'y': df_genre_grps.iloc[0],
              				'yaxis': 'y'
              			},
             			{
             				'name': 'news',
              				'type': 'bar',
              				'x': coln_labels[:10],
              				'xaxis': 'x2',
              				'y': df_genre_grps.iloc[1],
              				'yaxis': 'y2'
              			},
             			{
             				'name': 'social',
              				'type': 'bar',
              				'x': coln_labels[:10],
              				'xaxis': 'x3',
              				'y': df_genre_grps.iloc[2],
              				'yaxis': 'y3'
              			}
              		], 
              		
    		'layout': {
    					'template': 'plotly_dark',
               			'title': {'text': 'Top 10 Messages Categeries Groups as Direct, News and Social'},
               			'xaxis': {'anchor': 'y', 'domain': [0.0, 0.2888888888888889]},
               			'xaxis2': {'anchor': 'y2', 'domain': [0.35555555555555557, 0.6444444444444445]},
               			'xaxis3': {'anchor': 'y3', 'domain': [0.7111111111111111, 1.0]},
               			'yaxis': {'anchor': 'x', 'domain': [0.0, 1.0]},
               			'yaxis2': {'anchor': 'x2', 'domain': [0.0, 1.0]},
               			'yaxis3': {'anchor': 'x3', 'domain': [0.0, 1.0]}
               		}
        },
         
        {
    		'data': [
    					{
    						'name': 'related',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x',
              				'y': df_genre_grps[coln_labels[0]],
              				'yaxis': 'y'
              			},
             			{
             				'name': 'aid_related',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x2',
              				'y': df_genre_grps[coln_labels[1]],
              				'yaxis': 'y2'
              			},
             			{
             				'name': 'weather_related',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x3',
              				'y': df_genre_grps[coln_labels[2]],
              				'yaxis': 'y3'
              			},
             			{
             				'name': 'direct_report',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x4',
              				'y': df_genre_grps[coln_labels[3]],
              				'yaxis': 'y4'
              			},
             			{
             				'name': 'request',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x5',
              				'y': df_genre_grps[coln_labels[4]],
              				'yaxis': 'y5'
              			},
             			{
             				'name': 'other_aid',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x6',
              				'y': df_genre_grps[coln_labels[5]],
              				'yaxis': 'y6'
              			},
             			{
             				'name': 'food',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x7',
              				'y': df_genre_grps[coln_labels[6]],
              				'yaxis': 'y7'
              			},
             			{
             				'name': 'earthquake',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x8',
              				'y': df_genre_grps[coln_labels[7]],
              				'yaxis': 'y8'
              			},
             			{
             				'name': 'storm',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x9',
              				'y': df_genre_grps[coln_labels[8]],
              				'yaxis': 'y9'
              			},
             			{
             				'name': 'shelter',
              				'type': 'bar',
              				'x': df_genre_grps.index,
              				'xaxis': 'x10',
              				'y': df_genre_grps[coln_labels[9]],
              				'yaxis': 'y10'
              			}
              		], 
              		
    		'layout': {
    					'template': 'plotly_dark',
               			'title': {'text': 'Top 10 Message Categeories Grouped as Direct, News or Social'},
               			'xaxis': {'anchor': 'y', 'domain': [0.0, 0.16799999999999998]},
               			'xaxis10': {'anchor': 'y10', 'domain': [0.832, 1.0]},
               			'xaxis2': {'anchor': 'y2', 'domain': [0.208, 0.376]},
               			'xaxis3': {'anchor': 'y3', 'domain': [0.416, 0.584]},
               			'xaxis4': {'anchor': 'y4', 'domain': [0.624, 0.792]},
               			'xaxis5': {'anchor': 'y5', 'domain': [0.832, 1.0]},
               			'xaxis6': {'anchor': 'y6', 'domain': [0.0, 0.16799999999999998]},
               			'xaxis7': {'anchor': 'y7', 'domain': [0.208, 0.376]},
               			'xaxis8': {'anchor': 'y8', 'domain': [0.416, 0.584]},
               			'xaxis9': {'anchor': 'y9', 'domain': [0.624, 0.792]},
               			'yaxis': {'anchor': 'x', 'domain': [0.575, 1.0]},
               			'yaxis10': {'anchor': 'x10', 'domain': [0.0, 0.425]},
               			'yaxis2': {'anchor': 'x2', 'domain': [0.575, 1.0]},
               			'yaxis3': {'anchor': 'x3', 'domain': [0.575, 1.0]},
               			'yaxis4': {'anchor': 'x4', 'domain': [0.575, 1.0]},
               			'yaxis5': {'anchor': 'x5', 'domain': [0.575, 1.0]},
               			'yaxis6': {'anchor': 'x6', 'domain': [0.0, 0.425]},
               			'yaxis7': {'anchor': 'x7', 'domain': [0.0, 0.425]},
               			'yaxis8': {'anchor': 'x8', 'domain': [0.0, 0.425]},
               			'yaxis9': {'anchor': 'x9', 'domain': [0.0, 0.425]}
               			
               		}
		}
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
