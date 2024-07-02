import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from scipy.stats import norm

def load_data():
    titanic = fetch_openml("Titanic", version=1, as_frame=True)
    titanic.data = titanic.data.drop(columns=['home.dest', 'boat', 'body'], inplace=False, errors='ignore')

    bins = list(np.arange(0, 90, 10))
    titanic.data['age_group'] = pd.cut(titanic.data['age'], bins=bins, labels=bins[:-1])

    titanic.data['title'] = titanic.data['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    title_category = title_categories()
    titanic.data['title_categorical'] = titanic.data['title'].map(title_category)

    titanic.data['family_size'] = titanic.data['sibsp'] + titanic.data['parch'] + 1
    titanic.data['fare_per_person'] = titanic.data['fare'] / titanic.data['family_size']

    X_train, X_test, y_train, y_test = train_test_split(titanic['data'], titanic['target'], test_size=0.1,
                                                        random_state=42)
    X_train['survived'] = y_train
    X_test['survived'] = y_test

    X_train['is_alone'] = (X_train['family_size'] == 1).astype(int)
    X_test['is_alone'] = (X_test['family_size'] == 1).astype(int)

    X_train['multiple_cabins'] = X_train['cabin'].apply(
        lambda x: -1 if pd.isna(x) else (1 if len(x.split()) > 1 else 0))
    X_test['multiple_cabins'] = X_test['cabin'].apply(lambda x: -1 if pd.isna(x) else (1 if len(x.split()) > 1 else 0))

    imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
    X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.fit_transform(
        X_train[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']] = imputer.transform(
        X_test[['fare_per_person', 'sibsp', 'parch', 'pclass']])
    X_train['fare'] = X_train['fare'].fillna(X_train['fare_per_person'] * X_train['family_size'])
    X_test['fare'] = X_test['fare'].fillna(X_test['fare_per_person'] * X_test['family_size'])

    X_train['deck'] = X_train['cabin'].str[0]
    X_test['deck'] = X_test['cabin'].str[0]
    X_train['deck'] = X_train['deck'].replace('T', 'A')
    X_test['deck'] = X_test['deck'].replace('T', 'A')

    def encode_columns(df, columns):
        encoders = {}
        for col in columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            encoders[col] = le
        return df, encoders

    columns_to_encode = ['deck', 'age_group', 'title_categorical', 'sex', 'embarked']
    X_train, encoders = encode_columns(X_train, columns_to_encode)
    X_test, _ = encode_columns(X_test, columns_to_encode)

    for col in ['deck_encoded', 'age_group_encoded']:
        X_train[col] = X_train[col].replace(X_train[col].max() + 1, -1)
        X_test[col] = X_test[col].replace(X_test[col].max() + 1, -1)

    deck_imputer = KNNImputer(n_neighbors=2, missing_values=-1)
    X_train[['deck_encoded', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = deck_imputer.fit_transform(
        X_train[['deck_encoded', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']])
    X_test[['deck_encoded', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']] = deck_imputer.transform(
        X_test[['deck_encoded', 'pclass', 'fare_per_person', 'embarked_encoded', 'is_alone']])

    return X_train, y_train

def title_categories() -> dict:
    return {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Dr": "Dr", "Rev": "Rev",
            "Col": "Military", "Mlle": "Miss", "Major": "Military", "Ms": "Miss", "Mme": "Mrs", "Sir": "Nobility",
            "Capt": "Military", "Lady": "Nobility", "the Countess": "Nobility", "Jonkheer": "Nobility",
            "Don": "Nobility", "Dona": "Nobility"}

X_train, y_train = load_data()

file_path = 'C:/Users/kubac/PycharmProjects/titanic_projekt/titanic_projekt/updated_results_final.csv'
results_df = pd.read_csv(file_path)

results_df = results_df.dropna(subset=['cross_val_dict', 'accuracy_dict'])

results_df['cross_val_dict'] = results_df['cross_val_dict'].map(lambda x: f"{x:.4f}")
results_df['accuracy_dict'] = results_df['accuracy_dict'].map(lambda x: f"{x:.4f}")

results_df = results_df.sort_values(by='cross_val_dict', ascending=False)


top_10_classifiers = results_df.head(10)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, 'https://use.fontawesome.com/releases/v5.10.2/css/all.css'])


color_scheme = 'plasma'


numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
mu, std = norm.fit(X_train['age'].dropna())
x = np.linspace(0, 80, 100)
p = norm.pdf(x, mu, std) * len(X_train['age'].dropna()) * (80 / 30)

corr_matrix = X_train[numeric_cols].corr()
for i in range(len(corr_matrix)):
    corr_matrix.iloc[i, i] = 1

# Layout
app.layout = dbc.Container(
    fluid=True,
    style={'backgroundColor': '#00080', 'color': '#00080', 'fontFamily': 'Arial'},
    children=[
        dbc.Navbar(
            dbc.Container(
                [
                    dbc.NavbarBrand("Titanic Dashboard", style={'color': 'white', 'fontWeight': 'bold', 'fontSize': '24px'}),
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Overview", href="#", id="tab-overview-link")),
                            dbc.NavItem(dbc.NavLink("Prediction Results", href="#", id="tab-results-link")),
                        ],
                        className="ml-auto",
                        navbar=True
                    ),
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink(html.I(className="fab fa-linkedin"), href="https://www.linkedin.com", target="_blank")),
                            dbc.NavItem(dbc.NavLink(html.I(className="fab fa-facebook"), href="https://www.facebook.com", target="_blank")),
                            dbc.NavItem(dbc.NavLink(html.I(className="fab fa-instagram"), href="https://www.instagram.com", target="_blank")),
                        ],
                        className="ml-auto",
                        navbar=True
                    )
                ],
            ),
            color="dark",
            dark=True,
            className="mb-4"
        ),
        dbc.Tabs(id='tabs', active_tab='tab-overview', children=[
            dbc.Tab(label='Overview', tab_id='tab-overview', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-chart-bar"), " Correlation Matrix"], className='card-title'),
                                dcc.Graph(
                                    id='correlation-matrix',
                                    figure=px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                                     color_continuous_scale=color_scheme).update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font_color='white'
                                    ).update_traces(
                                        zmin=-1, zmax=1
                                    )
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-user"), " Age Distribution"], className='card-title'),
                                dcc.Graph(
                                    id='age-distribution-histogram',
                                    figure=px.histogram(X_train, x='age', nbins=30,
                                                        color_discrete_sequence=px.colors.sequential.Plasma)
                                    .update_traces(marker_line_width=1.5)
                                    .add_trace(go.Scatter(x=x, y=p, mode='lines', name='Gaussian Fit',
                                                          line=dict(color='orange')))
                                    .update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font_color='white'
                                    )
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-life-ring"), " Survival Analysis"], className='card-title'),
                                html.Label("Select analysis:"),
                                dcc.Dropdown(
                                    id='survival-analysis-dropdown',
                                    options=[
                                        {'label': 'Gender', 'value': 'sex'},
                                        {'label': 'Family Size', 'value': 'family_size'},
                                        {'label': 'Age Group', 'value': 'age_group'},
                                        {'label': 'Title Category', 'value': 'title_categorical'}
                                    ],
                                    value='sex'
                                ),
                                dcc.Graph(
                                    id='survival-by-category',
                                    config={'displayModeBar': False},
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-users"), " Family Size Distribution"], className='card-title'),
                                dcc.Graph(
                                    id='family-size-pie',
                                    figure=px.pie(X_train, names='family_size',
                                                  color_discrete_sequence=px.colors.sequential.Plasma).update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font_color='white'
                                    )
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=6),
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-id-badge"), " Title Categorical Distribution"], className='card-title'),
                                dcc.Graph(
                                    id='title-categorical-pie',
                                    figure=px.pie(X_train, names='title_categorical',
                                                  color_discrete_sequence=px.colors.sequential.Plasma).update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font_color='white'
                                    )
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=6)
                ])
            ]),
            dbc.Tab(label='Prediction Results', tab_id='tab-results', children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-table"), " Prediction Results"], className='card-title'),
                                dcc.Input(
                                    id='search-input',
                                    type='text',
                                    placeholder='Search...',
                                    style={'margin-bottom': '10px', 'width': '100%'}
                                ),
                                dash_table.DataTable(
                                    id='results-table',
                                    columns=[
                                        {"name": "Classifier", "id": "Classifier"},
                                        {"name": "Cross Validation Accuracy", "id": "cross_val_dict"},
                                        {"name": "Test Accuracy", "id": "accuracy_dict"},
                                        {"name": "Features", "id": "Features"}
                                    ],
                                    data=results_df.to_dict('records'),
                                    style_table={'overflowX': 'auto'},
                                    style_header={
                                        'backgroundColor': 'rgba(36,0,70,0.9)',
                                        'color': 'white',
                                        'fontWeight': 'bold'
                                    },
                                    style_cell={
                                        'backgroundColor': 'rgba(0,0,0,0.9)',
                                        'color': 'white'
                                    },
                                    sort_action='native',
                                    sort_mode='multi',
                                    page_action='native',
                                    page_current=0,
                                    page_size=10,
                                    filter_action='custom',
                                    filter_query=''
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card(
                            dbc.CardBody([
                                html.H3([html.I(className="fas fa-trophy"), " Top 10 Classifiers"], className='card-title'),
                                dcc.Graph(
                                    id='top-10-classifiers',
                                    figure=px.bar(
                                        top_10_classifiers,
                                        x='cross_val_dict',
                                        y='Classifier',
                                        title='Top 10 Classifiers by Cross Validation Accuracy',
                                        labels={'cross_val_dict': 'Cross Validation Accuracy'},
                                        orientation='h',
                                        color='cross_val_dict',
                                        color_continuous_scale=px.colors.sequential.Plasma
                                    ).update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font_color='white',
                                        yaxis={'categoryorder': 'total ascending'},
                                        xaxis={'automargin': True}
                                    )
                                )
                            ]),
                            style={"margin": "10px", "backgroundColor": "rgba(36,0,70,0.9)"}
                        ),
                    ], width=12)
                ])
            ])
        ])
    ]
)

@app.callback(
    Output('tabs', 'active_tab'),
    [Input('tab-overview-link', 'n_clicks'),
     Input('tab-results-link', 'n_clicks')]
)
def switch_tabs(n_clicks_overview, n_clicks_results):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'tab-overview'
    else:
        clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if clicked_id == 'tab-overview-link':
            return 'tab-overview'
        elif clicked_id == 'tab-results-link':
            return 'tab-results'

@app.callback(
    Output('survival-by-category', 'figure'),
    [Input('survival-analysis-dropdown', 'value')]
)
def update_survival_by_category(selected_category):
    if selected_category == 'sex':
        fig = px.histogram(X_train, x='sex', color='survived', barmode='group',
                           color_discrete_sequence=px.colors.sequential.Plasma)
    elif selected_category == 'family_size':
        fig = px.histogram(X_train, x='family_size', color='survived', barmode='group',
                           color_discrete_sequence=px.colors.sequential.Plasma)
    elif selected_category == 'age_group':
        fig = px.histogram(X_train, x='age_group', color='survived', barmode='group',
                           color_discrete_sequence=px.colors.sequential.Plasma)
    elif selected_category == 'title_categorical':
        fig = px.histogram(X_train, x='title_categorical', color='survived', barmode='group',
                           color_discrete_sequence=px.colors.sequential.Plasma)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title='Survival by ' + selected_category.capitalize()
    )
    return fig

@app.callback(
    Output('results-table', 'data'),
    Input('search-input', 'value')
)
def update_table(search_value):
    if search_value:
        filtered_df = results_df[results_df.apply(lambda row: row.astype(str).str.contains(search_value, case=False).any(), axis=1)]
    else:
        filtered_df = results_df
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)









