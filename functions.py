import pandas as pd
import numpy as np

from collections import defaultdict

import matplotlib.pylab as plt
import seaborn as sns

import datetime
import random
from tqdm import tqdm
import itertools

import networkx as nx
import queue
from collections import deque
from treelib import Node, Tree

import functions as fn

##################################################################################################################################
#############################################              RQ2               #####################################################
##################################################################################################################################

def set_marked_false (G):
    for node in G.nodes: 
        G.nodes[node]['marked'] = False
        

def rq2 (G, starting_page, num_clicks): 
    
    list_reachable_pages = []    
    q = deque()  # initialize the queue object 
    tree = Tree()
    new_tree = Tree()
    
    if not G.has_node(starting_page):
        print ('The node you inserted as starting point is not in the ')
        return 
    
    set_marked_false(G)
    
    G.nodes[starting_page]['marked'] = True
    q.append(starting_page)
    tree.create_node (starting_page, starting_page)
    
    while q:
        node = q.popleft()

        for neighbor_node in G.neighbors(node):
                
                if G.nodes[neighbor_node]['marked'] == False:

                    G.nodes[neighbor_node]['marked'] = True
                    q.append(neighbor_node)
                    tree.create_node (neighbor_node, neighbor_node, parent = node)
                    
                    
        if tree.depth() == num_clicks + 1:
            return new_tree
        
        new_tree = Tree(tree)  

##################################################################################################################################
#############################################              RQ3               #####################################################
##################################################################################################################################

def most_central_article(df, pages, final_category_dict, category, init_set_pages):
    
    in_degrees = []
    for page in init_set_pages:  #for every page in set_pages
        
        #counts how many links are pointing to that page 
        in_degrees.append(df.loc[df['target']==page]['edge'].count())
    
    most_central = init_set_pages[in_degrees.index(max(in_degrees))]
    
    return most_central


def iterations(set_pages, G, pages_reached, tot_pages_reached):
    set_pages = list(list(set(set_pages)-set(pages_reached)) + list(set(pages_reached)-set(set_pages)))
    pages_reached1 = []
    for p in pages_reached:
        try:
            pages_reached1.append([n for n in G.neighbors(p) if n in set_pages and n not in tot_pages_reached])
        except nx.exception.NetworkXError:
            pass
    
    pages_reached1 = set(list(itertools.chain.from_iterable(pages_reached1)))
    tot_pages_reached += list(pages_reached1)
    
    return pages_reached1, tot_pages_reached


def min_clicks(G, df, pages, final_category_dict,v,category, init_set_pages):
    i = 0 
    
    set_pages = init_set_pages
        
    pages_reached = [n for n in G.neighbors(v) if n in set_pages]
    tot_pages_reached = pages_reached
    
    print(f'Iteration {i}. Pages reached in {i} click: {len(pages_reached)}')
    
    while len(pages_reached) != 0:
        i += 1
        pages_reached, tot_pages_reached = iterations(set_pages, G, pages_reached, tot_pages_reached)
        print(f'Iteration {i}. Pages reached in {i} click: {len(pages_reached)}')
        
    if len(tot_pages_reached) == len(init_set_pages):
        return f'minimum number of clicks required to reach all pages: {i+1}'
    
    else:
        return i, f'Not Possible, total pages reached: {len(tot_pages_reached)} out of {len(init_set_pages)}'
    
    
    
    
    
'''This function checks, every time it is recalled, the neighbors of the page in input. It also differentiate between 
the pages that have to be reached, that are in the list of pages of the category in input, and the pages that 
are useful only to create a path but are not in the category of interest.
In output returns:
- the pages in the category of interest reached in that iteration
- the pages not in the category of interest reached in that iteration
- the updated list of pages visited, both from the category and not
- the updated list of pages visited ONLY of the category of interest'''

def iterations2(set_pages, G, pages_reached_in_setpages, pages_reached_not_in_set, tot_pages_reached, tot_pages_only_set):
    set_pages = list(list(set(set_pages)-set(pages_reached_in_setpages)) + list(set(pages_reached_in_setpages)-set(set_pages)))
    pages_reached_in_setpages1 = []   #neighbors that are in the set of pages to reach
    pages_reached_not_in_set1 = []    #all other neighbors
    
    tot = pages_reached_in_setpages +  pages_reached_not_in_set
    #for every page find the list of neighbors and differentiate between those in the category and those who are not
    for p in tot:
        neighbors = [n for n in G.neighbors(p)]
        try:
            pages_reached_in_setpages1.append([n for n in neighbors if n in set_pages and n not in tot_pages_reached])
            pages_reached_not_in_set1.append([n for n in neighbors if n not in set_pages and n not in tot_pages_reached])
        
        except nx.exception.NetworkXError:
            pass
    
    #combine all neighbors of all pages into one list
    pages_reached_in_setpages1 = list(set(list(itertools.chain.from_iterable(pages_reached_in_setpages1))))
    pages_reached_not_in_set1 = list(set(list(itertools.chain.from_iterable(pages_reached_not_in_set1))))
    
    #update the list of visited pages
    tot_pages_reached += pages_reached_in_setpages1 + pages_reached_not_in_set1
    tot_pages_only_set += pages_reached_in_setpages1
    
    return pages_reached_in_setpages1, pages_reached_not_in_set1, tot_pages_reached, tot_pages_only_set



'''Function that takes in input:
- the graph G
- the original dataframe with the information about the edges df
- the dictionary that has for keys the category and for values the list of pages in that category
- the starting page v
- the cateogry chosen in input by the user

The output is the minimum number of clicks needed to get from the central page v to the page that is further away from it
- I have to click to the max number of links to get there. If I cannot get to ALL the pages in the category from my initial
page, the algorithm stops and return the max number of pages it got before it had to stop'''

def min_clicks2(G, df, pages, final_category_dict,v,category, init_set_pages):
    i = 0 
     
    set_pages = init_set_pages                                      
    
    neighbors = [n for n in G.neighbors(v)]                                             #all the neighbors of node v
    pages_reached_in_setpages = [n for n in neighbors if n in set_pages]                #neighbors that are in the set of pages to reach
    pages_reached_not_in_set = list(set(neighbors) - set(pages_reached_in_setpages))    #all other neighbors
    
    tot_pages_reached = pages_reached_in_setpages + pages_reached_not_in_set            #tot pages already reached 
    tot_pages_only_set = pages_reached_in_setpages                                      #pages inside the category already reached
    
    print(f'Iteration {i}. Pages reached in {i+1} click: {len(pages_reached_in_setpages)}')
    
    while len(pages_reached_in_setpages) != 0:
        i += 1
        pages_reached_in_setpages, pages_reached_not_in_set, tot_pages_reached, tot_pages_only_set = iterations2(set_pages, G, pages_reached_in_setpages, 
                                                                                                                pages_reached_not_in_set, tot_pages_reached,
                                                                                                                tot_pages_only_set)
        print(f'Iteration {i}. Pages reached in {i+1} click: {len(pages_reached_in_setpages)}')
        
    if len(tot_pages_reached) == len(init_set_pages):                                   #if I visit all the pages in input
        return f'minimum number of clicks required to reach all pages: {i+1}'           #I return the maximum number of clicks to reach the page that is further than v
    
    #if the sum is not equal it means that there is at least one page that cannot be reached from v
    else:
        return i, f'Not Possible, total pages reached: {len(tot_pages_only_set)} out of {len(init_set_pages)}'
    
##################################################################################################################################
#############################################              RQ4               #####################################################
##################################################################################################################################

def hyper_remove(H, df, set_pages, final_category_dict , category1 , category2, u, v, counter):
    i = 0 
    
    try:
        #page reached can be only in the subgraph
        pages_reached = [n for n in H.neighbors(v) if n in set_pages]
        tot_pages_reached = pages_reached
        
        #check if v and u are neighbours 
        if u in H.neighbors(v):    
            print(f'Path #{counter}: Shortest distance {i+1}')
            counter += 1
            #if v and u are neighbour we erase the edge (v,u)
            df = df[(df['source'] != v) & (df['target'] != u)]
        
        #function keep going until has no more page to reach
        while len(pages_reached) != 0:
            i += 1
            pages_reached, tot_pages_reached = iterations(set_pages, H, pages_reached, tot_pages_reached)
            
            #for every reached page we check if its neighbour is u
            for x in pages_reached:
                if u in H.neighbors(x):
                    counter += 1
                    print(f'Path #{counter}: Shortest distance {i+1}')
                    #removing the hyperlink
                    df = df[(df['source'] != x) & (df['target'] != u)]
                    #run again the function  with the new edge dataset passing the current counter of hyperlinks removed
                    hyper_removed = hyper_remove(H, df, new_set_pages, final_category_dict , category1 , \
                                                 category2, u, v, counter)  

    except TypeError:
        pass
    
    hyper_removed = counter
    if hyper_removed != 0:
        return hyper_removed, f'Number of minimum hyperlink removed to disconnect {u} and {v}'
    
    else:
        print(f'{v} and {u} are not connected by default')



##################################################################################################################################
#############################################              RQ5               #####################################################
##################################################################################################################################

def distance(G, df, v,  set_pages, page_list_temp):
    
    i = 1 
    end = False
    pages_reached = [n for n in G.neighbors(v) if n in set_pages]
    tot_pages_reached = pages_reached
    
    #check if v is directly connected with category nodes in page_list_temp
    for q in G.neighbors(v):
        if q in page_list_temp:
            end = True
    
    #end the loop either if there are no pages left or function reach the category
    while len(pages_reached) != 0 and end != True :
            i += 1
            pages_reached, tot_pages_reached = iterations(set_pages, G, pages_reached, tot_pages_reached)
            #for every page reached check if its neighbour are part of category nodes in page_list_temp
            for x in pages_reached:
                for q in G.neighbors(x):
                    if q in page_list_temp:
                         end = True
                            
    #if category is not found let's put the distance as 100000 (high number) and it means node is not linked
    if end == True:
        return i
    else:
        return 100000
    




    def category_distance(G, df, final_category_dict, C0, categories):
    
    #dictionary where the distances will be saved
    dist_from_C0 = {}
    
    #nodes in C0
    page_list0 = final_category_dict[C0]
    #All nodes in G
    set_pages = list(set(G.nodes))
    
    #for very category 
    for i in categories:
        page_list_temp = final_category_dict[i]
        dist_from_C0[i] = [] 
        
        #for every node in C0
        for x in page_list0:    
            #here is computed the distance between the node in C0 and the category i
            dist = distance(G, df, x , set_pages, page_list_temp)
            print(dist)        
            dist_from_C0[i].append(dist) 
            
        print(f' category: {i} completed')
    
    #converting dictionary in a DataFrame 
    data_dist = pd.concat({key: pd.Series(value) for key, value in dist_from_C0.items()}, axis=1) 
    
    return data_dist





##################################################################################################################################
#############################################              RQ6               #####################################################
##################################################################################################################################

def nodes_to_category(G1, final_category_dict):
    cat = []
    for node in tqdm(G1.nodes()):
        for key, value in final_category_dict.items():
            if node in value:
                cat.append(key)
                break

    node_to_cat = defaultdict()
    for i,node in tqdm(enumerate(G1.nodes())):
        node_to_cat[node]  = cat[i]
        
    return node_to_cat


def count_edges_per_category(G,inverted_link2):
    dict_edge_cat = {}

    for i,source in tqdm(enumerate(G.nodes())):
        nodes_linked = [elem for elem in G[source]]   #list of nodes that can be reached from that node

        source_cat = inverted_link2[source][0]
        if source_cat not in  dict_edge_cat.keys():
            dict_edge_cat[source_cat] = {}


        for target in nodes_linked:
            target_cat = inverted_link2[target][0]
            if target_cat not in dict_edge_cat[source_cat].keys():
                dict_edge_cat[source_cat][target_cat] = 1

            else:   
                dict_edge_cat[source_cat][target_cat] +=1
    return dict_edge_cat


def get_transition_matrix(df4, G1):
    M = np.zeros((len(G1.nodes()),len(G1.nodes())))
    for i in tqdm(range(len(M))):
        s = df4.iloc[i][-1]
        for j in range(len(M)):
            M[i][j] = df4.iloc[i][j]/s
    return M


def final_pagerank_output(G1,v):
    cat_to_pagerank = defaultdict()
    for i, node in enumerate(G1.nodes()):
        cat_to_pagerank[node] = v[0][i]
    x = sorted(cat_to_pagerank, key=cat_to_pagerank.get)
    x.reverse()
    return x


def pagerank(n_iter, a, nodes, M, G1):
    n = len(M)               #number of nodes
    m_id = np.ones((n, n))   #matrix n*n with all ones 

    P = (1/n)*a*m_id + (1-a)*M              #probability to get to one node to the other
 
    starting_page = random.choice(nodes)    #select a random page where to start random
    v = np.zeros((1,len(nodes)))         
    v[0][nodes.index(starting_page)] = 1.0  #create a strating vector that has 1 only in the position of the page selected

    for i in range(n_iter):                 #repeat the process 100 time 
        v = v @ P
        
    return final_pagerank_output(G1,v)
    


########Another approach functions


    def retrieve_M (G1, a, b, row_of_zeroes ):

    n = int(len(G1.nodes())/7)
    M = np.zeros((n, n), dtype = np.float32 )  #initialize a matrix nxn (n = #nodes)
    nodes = [node for node in G1.nodes()]             #list of nodes in our graph

    #for each row that corresponds to each node
    for i,source in tqdm(enumerate(G1.nodes())):
        
        
        nodes_linked = [elem for elem in G1[source]]   #list of nodes that can be reached from that node
        for target in nodes_linked:
            
            if (i >= a*n) and (i<(a+1)*n):
                ind = nodes.index(target)
                
                if (ind >= b*n) and (ind <(b+1)*n):
                    M[i-n*a][ind-n*b] = G1[source][target][0]['weight']  #I assign the weight that I previously calculated
                                                                         #in the correct cell of the matrix
                        
                if i in row_of_zeroes:
                    M[i-n*a] = round(1/(n*7),6) 
        
    return M
                




def pagerank2(n_iter, alpha, G1, row_of_zeroes):
    
    starting_page = random.choice(nodes)    #select a random page where to start random
    v = np.zeros(len(nodes))       
    v[nodes.index(starting_page)] = 1.0   #create a strating vector that has 1 only in the position of the page selected
    n = int(len(G1.nodes())/7)
    v = np.reshape(v, (7, n))      #the vector is split as well in seven parts
    
    v_temp = v    #copy of v to host tempporarly the results
    
    v_final = np.zeros(len(nodes)) 
    v_final = np.reshape(v_final, (7, n))
    
    
    for b in range(7):
        for a in range(7):
            M = retrieve_M(G1, a, b, row_of_zeroes )   #the submatrix M we are working with each time
            v_temp[a] = v[a]
            
            for i in range(n_iter):                     # number of iterations
                n = len(M)                                #number of nodes taken
                P = (1/n)*alpha + (1-alpha)*M
                v_temp[a] = v_temp[a] @ P            
                
            v_final[b] += v_temp[a] 

    v_final = np.reshape(v_final, (1, n*7))
    
    return v                                #the final vector v will contain the probbaility of being at page i
                      


def order_categories(v, nodes, final_category_dict):
    pagerank = defaultdict()
    for category, list_of_pages in final_category_dict.items():
        pagerank[category] = sum([v[nodes.index[page]] for page in list_of_pages])
        
    sorted_dict = {key: v for key, value in sorted(pagerank.items(), key=lambda item: item[1])}
    
    return sorted_dict.keys()