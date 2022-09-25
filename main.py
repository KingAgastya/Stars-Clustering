import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

rows = []
with open('cleaned_data.csv', 'r') as f:
    csv_r = csv.reader(f)
    for i in csv_r:
        rows.append(i)

headers = rows[0]
stars_data = rows[1:]
headers[0] = 'Index'

star_data = []
for star in stars_data:
    if star[3] != '?': 
        star[3] = float(star[3].strip("\'"))*1.989e+30
    
    if star[4] != '?':
        star[4] = float(star[4].strip("\'"))*6.957e+8
    star_data.append(star)

star_data_gravity = []
for star in star_data:
    if star[3] != '?' and star[4] != '?':
        gravity = (6.674e-11 * float(star[3]))/(float(star[4])*float(star[4]))
    star.append(gravity)
    star_data_gravity.append(star)

star_data_mass = []
for star in star_data:
    if star[3] != '?':
        mass = float(star[3])
    star.append(mass)
    star_data_mass.append(star)

star_data_radius = []
for star in star_data:
    if star[4] != '?':
        radius = float(star[4])
    star.append(radius)
    star_data_radius.append(star)

star_data_gravity.sort()
star_data_mass.sort()
star_data_radius.sort()

plt.plot(star_data_mass, star_data_radius)
plt.title('Graph of Mass-Radius')
plt.xlabel('Mass')
plt.ylabel('Radius')
plt.show()


plt.plot(star_data_mass, star_data_gravity)
plt.title('Graph of Mass-Gravity')
plt.xlabel('Mass')
plt.ylabel('Gravity')
plt.show()

X = []

for index, planet_mass in enumerate(star_data_mass):
  temp_list = [
                star_data_radius[index], 
                planet_mass
              ]
  X.append(temp_list)


wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)

plt.figure(figsize = (10, 5))
sns.lineplot(range(1, 11), wcss, marker = 'o', color = 'red')
plt.title("The Elbow Method")
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

name = []
distance = []
mass = []
radius = []
gravity = []

for i in star_data_gravity:
    name.append(i[1])
    distance.append(i[2])
    mass.append(i[3])
    radius.append(i[4])
    gravity.append(i[5])

df = pd.DataFrame(
    list(zip(name, distance, mass, radius, gravity)),
    columns=["Star Name", "Distance", "Mass", "Radius", "Gravity"],
)
df.to_csv('data.csv')