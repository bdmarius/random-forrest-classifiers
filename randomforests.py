from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
import pydotplus
from sklearn import tree

def getWeather():
    return ['Clear', 'Clear', 'Clear', 'Clear', 'Clear', 'Clear',
            'Rainy', 'Rainy', 'Rainy', 'Rainy', 'Rainy', 'Rainy',
            'Snowy', 'Snowy', 'Snowy', 'Snowy', 'Snowy', 'Snowy']

def getTimeOfWeek():
    return ['Workday', 'Workday', 'Workday',
            'Weekend', 'Weekend', 'Weekend',
            'Workday', 'Workday', 'Workday',
            'Weekend', 'Weekend', 'Weekend',
            'Workday', 'Workday', 'Workday',
            'Weekend', 'Weekend', 'Weekend']

def getTimeOfDay():
    return ['Morning', 'Lunch', 'Evening',
            'Morning', 'Lunch', 'Evening',
            'Morning', 'Lunch', 'Evening',
            'Morning', 'Lunch', 'Evening',
            'Morning', 'Lunch', 'Evening',
            'Morning', 'Lunch', 'Evening',
            ]

def getTrafficJam():
    return ['Yes', 'No', 'Yes',
            'No', 'No', 'No',
            'Yes', 'Yes', 'Yes',
            'No', 'No', 'No',
            'Yes', 'Yes', 'Yes',
            'Yes', 'No', 'Yes'
            ]

def printTree(classifier, index):
    feature_names = ['Weather', 'Time of Week', 'Time of Day']
    target_names = ['Yes', 'No']
    # Build the daya
    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names)
    # Build the graph
    graph = pydotplus.graph_from_dot_data(dot_data)

    # Show the image
    Image(graph.create_png())
    graph.write_png("tree" + str(index) + ".png")

# Get the data
weather = getWeather()
timeOfWeek = getTimeOfWeek()
timeOfDay = getTimeOfDay()
trafficJam = getTrafficJam()

labelEncoder = preprocessing.LabelEncoder()

# Encode the features and the labels
encodedWeather = labelEncoder.fit_transform(weather)
encodedTimeOfWeek = labelEncoder.fit_transform(timeOfWeek)
encodedTimeOfDay = labelEncoder.fit_transform(timeOfDay)
encodedTrafficJam = labelEncoder.fit_transform(trafficJam)

# Build the features
features = []
for i in range(len(encodedWeather)):
    features.append([encodedWeather[i], encodedTimeOfWeek[i], encodedTimeOfDay[i]])

classifier = RandomForestClassifier(n_estimators=5)
classifier.fit(features, encodedTrafficJam)

# ["Snowy", "Workday", "Morning"]
print(classifier.predict([[2, 1, 2]]))
# Prints [1], meaning "Yes"
# ["Clear", "Weekend", "Lunch"]
print(classifier.predict([[0, 0, 1]]))
# Prints [0], meaning "No"

for index in range(len(classifier.estimators_)):
    printTree(classifier.estimators_[index], index)