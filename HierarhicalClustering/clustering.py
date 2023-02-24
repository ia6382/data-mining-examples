import csv
import math
from matplotlib import pyplot as plt 

start_year = 2000 #min 1975
end_year = 2010 #max 2019

years = {} #for normalizing euclidian distance

def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    #create lookup dict to get index for each country in that year
    #also get names of all coutnries for DATA
    #global years_and_countries = {}

    index = 0
    countries_and_years = {}
    countries = set()
    f = open(file_name, "rt", encoding="utf8")
    next(f) #skip header
    for line in csv.reader(f):
        year = line[0]
        from_country = line[2]
        to_country = line[3]
        key = year+to_country #mogoce to_country zaradi napake v letu 2003 ceprav je lepse from_country
        
        year = int(year)
        if year >= start_year and year <= end_year:
            #if not(from_country in countries):
            countries.add(from_country)

            if not(key in countries_and_years):
                countries_and_years[key] = index
                years[index] = year
                index += 1

    #create DATA
    DATA = {}
    n = len(countries_and_years)
    for i in countries:
        DATA[i] = [float('nan')]*n

    #update profiles in DATA
    f = open(file_name, "rt", encoding="utf8")
    next(f) #skip header
    for line in csv.reader(f):
        #parse line
        year = line[0]
        voting = line[1]
        from_country = line[2]
        to_country = line[3]
        points = int(line[4])
        key = year+to_country

        year = int(year)
        if year >= start_year and year <= end_year:
            if voting == "J":
                index = countries_and_years[key]
                if math.isnan(DATA[from_country][index]):
                    DATA[from_country][index] = points
                else:
                    DATA[from_country][index] += points

    #print(DATA)
    return DATA

class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]
        self.distances = {}

        #self.years = years

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """
        #euclidian distance
        val1 = self.data[r1]
        val2 = self.data[r2]

        d = 0
        in_common = False
        for i,j in zip(val1,val2):
            if math.isnan(i) or math.isnan(j):
                continue
            else:
                in_common = True
                d += (i - j)**2

        if in_common == False: #rows have nothing in common
            return float("nan")
        else:
            """
            #weigh the distance with number of apperances
            apperances = 0
            year = 0
            for (i,(v1,v2)) in enumerate(zip(val1,val2)):
                if years[i] > year and not(math.isnan(v1)) and not(math.isnan(v2)):
                    year = years[i]
                    apperances += 1   
            d = d/apperances
            """
            d = math.sqrt(d)
            return d

    def flatten_list(self, c):
        if len(c) < 2:
            return [c[0]]
        else:
            return self.flatten_list(c[0]) + self.flatten_list(c[1])

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        flat_c1 = self.flatten_list(c1)
        flat_c2 = self.flatten_list(c2)
        """
        #single linkage
        distance = float('inf')
        for i in flat_c1:
            for j in flat_c2:
                d = self.row_distance(i, j)
                if math.isnan(d): #if distance is nan it should not effect linkage
                    continue
                if d < distance:
                    distance = d
        #if two groups have nothing in common (all distances are nan) they shouldnt affect each other
        if math.isinf(distance):
            distance = float('nan')
        """
        #complete linkage
        distance = float('-inf')
        for i in flat_c1:
            for j in flat_c2:
                d = self.row_distance(i, j)
                if math.isnan(d): #if distance is nan it should not effect linkage
                    continue
                if d > distance:
                    distance = d
        #if two groups have nothing in common (all distances are nan) they shouldnt affect each other
        if math.isinf(distance):
            distance = float('nan')
        """
        #average linkage
        len_c1 = len(flat_c1)
        len_c2 = len(flat_c2)

        distance_sum = 0
        for i in flat_c1:
            for j in flat_c2:
                d = self.row_distance(i, j)
                if math.isnan(d): #if distance is nan it shouldnt effect linkage -> shorten the lenght of groups
                    if len_c1 > len_c2:
                        len_c1 -= 1
                    else:
                        len_c2 -= 1
                    continue
                
                distance_sum += d
        #if two groups have nothing in common (all distances are nan) they shouldnt affect each other
        if len_c1 < 1 or len_c2 < 1:
            distance = float('nan')
        else:
            distance = distance_sum / (len_c1*len_c2)        
        """

        return distance

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        minD = float('inf')
        c1, c2 = None, None 
        id1, id2 = 0, 0 #indeksa najblizjih gruc c1 in c2
        for indx1, i in enumerate(self.clusters[:-1]):
            for indx2, j in enumerate(self.clusters[(indx1+1):], start=indx1+1):
                D = self.cluster_distance(i, j)
                if math.isnan(D): #if two clusters have nothing in common they are ignored
                    continue
                if(D < minD):
                    minD = D
                    c1 = i
                    c2 = j
                    id1 = indx1
                    id2 = indx2

        #max in min za indeksa je zato da se ne porusi zaporedje indeksov, ko kasneje brisemo elemente
        return [max(id1, id2), min(id1, id2), [c1, c2], minD]

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """
        while len(self.clusters) > 2:
            [id1, id2, closestPair, dist] = self.closest_clusters()

            del self.clusters[id1]
            del self.clusters[id2]
            self.clusters.append(closestPair)

            #save cluster merge distance
            self.distances[str(closestPair)] = dist

        #one more iteration for last pair
        [id1, id2, closestPair, dist] = self.closest_clusters()
        self.distances[str(closestPair)] = dist

    def cut_groups(self, distance):

        def recursive_cut(c):
            if len(c)<2: #osamelec
                return [c]
            elif self.distances[str(c)] <= distance:
                return [c]
            else:
                return recursive_cut(c[0]) + recursive_cut(c[1])
        
        return recursive_cut(self.clusters)

    def analyze_groups(self, file_name, distance):
        groups = self.cut_groups(distance)

        #get DATA for calculating the averages for groups later on

        #create lookup dict to get index for each country
        #also get names of all coutnries and the number of their apperances
        index = 0
        countries = {}
        country_apperances = {}
        countries_and_years = set() #to make sure we arent counting apperances for the same year
        f = open(file_name, "rt", encoding="utf8")
        next(f) #skip header
        for line in csv.reader(f):
            year = line[0]
            from_country = line[2]
            to_country = line[3]
            key = year + from_country
            year = int(year)

            if year >= start_year and year <= end_year:
                if not(from_country in country_apperances):
                    country_apperances[from_country] = 1
                    countries_and_years.add(key)
                elif not(key in countries_and_years):
                    countries_and_years.add(key)
                    country_apperances[from_country] += 1

                if not(to_country in countries):
                    countries[to_country] = index
                    index += 1

        #create DATA
        DATA = {}
        n = len(countries)
        for i in country_apperances:
            DATA[i] = [0]*n

        #update profiles in DATA
        f = open(file_name, "rt", encoding="utf8")
        next(f) #skip header
        for line in csv.reader(f):
            #parse line
            year = int(line[0])
            voting = line[1]
            from_country = line[2]
            to_country = line[3]
            points = int(line[4])

            if year >= start_year and year <= end_year:
                index = countries[to_country]
                DATA[from_country][index] += points


        #calculate average for each group from DATA
        flat_groups=[]
        for group in groups:
            flat_groups.append(self.flatten_list(group))

        for group in flat_groups:
            avg_points = [0]*n
            apperances = 0
            
            #sum the points for each country
            for country in group:
                for i, p in enumerate(DATA[country]):
                    avg_points[i] += p
               
                apperances += country_apperances[country]
            
            #weighted average
            for i, p in enumerate(avg_points):
                avg_points[i] = round(p/apperances, 2)

            #print user friendly results
            #ordered list from dict
            list_countries = []
            for (key, val) in countries.items():
                list_countries.append((key, val))
            list_countries = sorted(list_countries, key=lambda x: x[1])

            #join names with points and order
            ordered_avg = []
            for (p, (k, v)) in zip(avg_points, list_countries):
                ordered_avg.append((p, k))
            ordered_avg = sorted(ordered_avg, key=lambda x: x[0])

            print("GROUP: "+str(group)) 
            print("AVERAGE VOTES: ")
            print(ordered_avg)
            print("---------------------------------------------\n")

                
    def plot_tree(self):
        """    
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        def recursive_print(c, depth):
            if len(c) < 2:
                print("    "*depth+"----"+c[0])
            else:
                recursive_print(c[0], depth+1)
                print("    "*depth+"----|"+str(round(self.distances[str(c)], 2)))
                recursive_print(c[1], depth+1)

        recursive_print(self.clusters, 1)

    def plot_tree_graphics(self):
        #plot using matplot lib
        x_start = min(self.distances.values())-1

        def recursive_plot(c, order, prevDist):
            if len(c) < 2:
                plt.plot([x_start, prevDist], [order, order], color="g") 
                plt.text(x_start, order, c[0], fontsize=9)
                return (order, order)
            else:
                dist = round(self.distances[str(c)], 2)

                (order1, y1) = recursive_plot(c[0], order, dist)
                (order2, y2) = recursive_plot(c[1], order1+2, dist)
                plt.plot([dist, prevDist],[order1+1, order1+1], color="g")
                plt.plot([dist, dist],[y1, y2], color="g")
                return (order2, order1+1)

        recursive_plot(self.clusters, 1, round(self.distances[str(self.clusters)]+2))
        plt.show()


if __name__ == "__main__":
    DATA_FILE = "eurovisionData.csv"
    DATA = read_file(DATA_FILE)
    hc = HierarchicalClustering(DATA)
    hc.run()
    #hc.plot_tree()
    hc.analyze_groups(DATA_FILE, 21.5)
    hc.plot_tree_graphics()