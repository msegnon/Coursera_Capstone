from sklearn.cluster import KMeans
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import requests
# convert an address into latitude and longitude values
from geopy.geocoders import Nominatim

import numpy as np
import json  # library to handle JSON files


import requests  # library to handle requests
# tranform JSON file into a pandas dataframe
from pandas.io.json import json_normalize
import folium  # map rendering library
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
# Define Foursquare credentials and version
CLIENT_ID = 'OXYVQITI2O451AABHJZKG5MWUZH2QGL0L2R0S43T5ZJBBEN0'  # your Foursquare ID
# your Foursquare Secret
CLIENT_SECRET = 'AGGLVHKVJZESGHN45XI0TBWABEYC0ACAE12K5YU0MEP3HJYP'
VERSION = '20180605'  # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

#Function to sort venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


def print_n_top_venues(num_top_venues, data):
    for hood in data['Neighbourhood']:
        print("----"+hood+"----")
        temp = data[data['Neighbourhood'] == hood].T.reset_index()
        temp.columns = ['venue', 'freq']
        temp = temp.iloc[1:]
        temp['freq'] = temp['freq'].astype(float)
        temp = temp.round({'freq': 2})
        print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
        print('\n')


def return_one_hot_encoding ( feature, data, add_feature ):
    onehot = pd.get_dummies(data[[feature]], prefix="", prefix_sep="")

    # add neighborhood column back to dataframe
    onehot[add_feature] = data[add_feature]

    # move neighborhood column to the first column
    fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
    onehot = onehot[fixed_columns]
    
    return onehot


def getNearbyVenues(names, latitudes, longitudes, radius=500):

    venues_list = []
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)

        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID,
            CLIENT_SECRET,
            VERSION,
            lat,
            lng,
            radius,
            100)

        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']

        # return only relevant information for each nearby venue
        venues_list.append([(
            name,
            lat,
            lng,
            v['venue']['name'],
            v['venue']['location']['lat'],
            v['venue']['location']['lng'],
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame(
        [item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood',
                             'Neighbourhood Latitude',
                             'Neighbourhood Longitude',
                             'Venue',
                             'Venue Latitude',
                             'Venue Longitude',
                             'Venue Category']

    print('Found {} venues in {} neighbourhoods.'.format(
        nearby_venues.shape[0], len(venues_list)))

    return(nearby_venues)

def get_geographical_coordinate ( address ):
    geolocator = Nominatim( user_agent="to_explorer" )
    location = geolocator.geocode( address )
    return location

def print_borough_and_neighbourhoods ( city, city_data ):
    print('{} has {} boroughs and {} neighborhoods.'.format(
        city, len(city_data['Borough'].unique()), city_data.shape[0]))


def create_map(data, latitude, longitude, zoom_start):
    # create map of New York using latitude and longitude values
    map_city = folium.Map(location=[latitude, longitude], zoom_start=10)
    
    # add markers to map
    for lat, lng, borough, neighbourhood in zip(data['Latitude'], data['Longitude'], data['Borough'], data['Neighbourhood']):
        label = '{}, {}'.format(neighbourhood, borough)
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            parse_html=False).add_to(map_city)

    return map_city

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']

    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

def create_dataframe_of_top_n_venues(num_top_venues, data):
    indicators = ['st', 'nd', 'rd']

    # create columns according to number of top venues
    columns = ['Neighbourhood']
    for ind in np.arange(num_top_venues):
        try:
            columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
        except:
            columns.append('{}th Most Common Venue'.format(ind+1))

        # create a new dataframe
    neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
    neighbourhoods_venues_sorted['Neighbourhood'] = data['Neighbourhood']

    for ind in np.arange( data.shape[0] ):
        neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(data.iloc[ind, :], num_top_venues)

    return neighbourhoods_venues_sorted


def create_dataframe_including_cluster_and_top_n_venues (cluster_labels, data, neighborhoods_venues_sorted):
    data_merged = data

    # add clustering labels
    data_merged['Cluster Labels'] = cluster_labels

    # merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
    data_merged = data_merged.join( neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

    return data_merged


def create_map_clusters(kclusters, latitude, longitude, data_merged,zoom_start):
    # create map
    map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

    # set color scheme for the clusters
    x = np.arange(kclusters)
    ys = [i+x+(i*x)**2 for i in range(kclusters)]
    colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
    rainbow = [colors.rgb2hex(i) for i in colors_array]

    # add markers to the map
    #markers_colors = []
    for lat, lon, poi, cluster in zip(data_merged['Latitude'], data_merged['Longitude'], data_merged['Neighbourhood'], data_merged['Cluster Labels']):
        label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            popup=label,
            color=rainbow[cluster-1],
            fill=True,
            fill_color=rainbow[cluster-1],
            fill_opacity=0.7).add_to(map_clusters)

    return map_clusters


def return_inertia(n_clusters, X):
    km = KMeans(n_clusters=n_clusters, init='k-means++',max_iter=15, random_state=8)
    km.fit(X)
    return km.inertia_

def return_labels( n_clusters, X, random_state=0):
    # run k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit( X )

    # check cluster labels generated for each row in the dataframe
    return kmeans.labels_


from yellowbrick.cluster import KElbowVisualizer
def find_optimal_clusters ( X ):
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 21))

    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show()


def load_and_process_newyork_data ( url ):
    myfile = requests.get(url)
    open('newyork_data.json', 'wb').write(myfile.content)

    with open('newyork_data.json') as json_data:
        ny_data = json.load(json_data)


    neighbourhoods_data = ny_data['features']

    # define the dataframe columns
    column_names = ['Borough', 'Neighbourhood', 'Latitude', 'Longitude']

    # instantiate the dataframe
    neighbourhoods = pd.DataFrame(columns=column_names)

    for data in neighbourhoods_data:
        borough = neighbourhood_name = data['properties']['borough']
        neighbourhood_name = data['properties']['name']

        neighbourhood_latlon = data['geometry']['coordinates']
        neighbourhood_lat = neighbourhood_latlon[1]
        neighbourhood_lon = neighbourhood_latlon[0]

        neighbourhoods = neighbourhoods.append({'Borough': borough,
                                            'Neighbourhood': neighbourhood_name,
                                            'Latitude': neighbourhood_lat,
                                            'Longitude': neighbourhood_lon}, ignore_index=True)
    return neighbourhoods


def load_and_process_toronto_data ( url ):
    #Obtain Postal Code, Borough, and Neighborhood information from Wikipedia
    df_toronto = pd.read_html( url , header=0)[0]
    df_toronto = df_toronto[df_toronto['Borough'] != "Not assigned"]
    geo_data = pd.read_csv("Geospatial_Coordinates.csv")
    toronto_data = pd.merge(df_toronto, geo_data, how='left', on='Postal Code')

    return toronto_data


def select_borough ( city_data, borough_name ):
    borough_data = city_data[city_data['Borough'] == borough_name].reset_index(drop=True)

    return borough_data
    
