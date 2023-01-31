#Colors
color1 = (148/255.0,  53/255.0,  65/255.0)
color2 = (80/255.0, 132/255.0, 167/255.0)
color3 = (81/255.0, 196/255.0, 112/255.0)
color4 = (20/255.0, 100/255.0, 50/255.0)
color5 = (50/255.0, 10/255.0, 139/255.0)


##UniformDist dataset
import numpy as np
import matplotlib.pyplot as plt

# creating the dataset	
#data = {'3DRoad':12.02, 'Porto':496.6, '3DIono':58,'KITTI':196.3}
#Impact of DS size
data = {'100K': 3.5, '200K': 3.25, '400K': 4.28, '800K': 4.15, '1M': 2.43}
#99th Percentile
data = {'100K': 1.5, '200K': 1.23, '400K': 1.7, '800K': 1.78, '1M': }
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (5, 5))
 
# creating the bar plot
plt.bar(courses, values, color = color1, align = 'center',
        width = 0.4)
 
plt.xlabel("Dataset size")
plt.ylabel("Speedup")
#plt.title("Students enrolled in different courses")
plt.show()

save_file_name = "Plots/uniform_dist_99th_speedup"
plt.savefig(save_file_name)
plt.show()
plt.clf()


'''
#Bargraph wih threshold line
import numpy as np
import matplotlib.pyplot as plt

# some example data
threshold = 1
data = {'3DRoad':12.02, 'Porto':496.6, '3DIono':58,
        'KITTI':196.3}
x = data.keys()

# split it up
above_threshold = np.array([1,1,1,1])
below_threshold = np.array([12.02,496.6,58,196.3])

# and plot it
fig, ax = plt.subplots()
ax.bar(x, below_threshold, 0.35, color="g")
ax.bar(x, above_threshold, 0.35, color="r",
        bottom=below_threshold)

# horizontal line indicating the threshold
ax.plot([0., 4.5], [threshold, threshold], "k--")

fig.savefig("k=5.png")'''




'''#Impact of k
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize =(8, 6))
 
# impact of k
h_3droad = [12.02,6.26]
h_porto = [496.6,99.9]
h_3diono = [58,76.43]
h_kitti = [196.3,71.3]

 
# Set position of bar on X axis
br1 = np.arange(len(h_kitti))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
 
# Make the plot
plt.bar(br1, h_3droad, color = color1, width = barWidth,
        edgecolor ='grey', label ='3DRoad')
plt.bar(br2, h_porto, color = color2, width = barWidth,
        edgecolor ='grey', label ='Porto')
plt.bar(br3, h_3diono, color = color3, width = barWidth,
        edgecolor ='grey', label ='3DIono')
plt.bar(br4, h_kitti, color = color4, width = barWidth,
        edgecolor ='grey', label ='KITTI')
 
# Adding Xticks
plt.xlabel('k', fontsize = 10)
plt.ylabel('Speedup', fontsize = 10)
plt.xticks([r + barWidth for r in range(len(h_kitti))],
        ['5', '660'])
 
plt.legend()
plt.savefig("Plots/ImpactOfK.pdf")'''




'''#Dataset size 1M
import numpy as np
import matplotlib.pyplot as plt

#Colors
color1 = (148/255.0,  53/255.0,  65/255.0)
color2 = (80/255.0, 132/255.0, 167/255.0)
color3 = (81/255.0, 196/255.0, 112/255.0)
color4 = (20/255.0, 100/255.0, 50/255.0)
color5 = (50/255.0, 10/255.0, 139/255.0)
 
# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize =(8, 6))
 
# 99th percentile
ds_100k = [23.8, 11.4, 14.5]
ds_200k = [33.7, 11.5, 18.3]
ds_400k = [47.1, 13.9, 21]
ds_800k = [61.1, 20, 22.6]

# DS size
ds_100k = [7.12, 28.6, 32.5, 37.7]
ds_200k = [6.13, 44.2, 54.6, 53.7]
ds_400k = [6.26, 99.9, 76.4, 71.3]
ds_800k = [0, 65.1, 73.6, 209.7]
ds_1000k = [0, 121.3, 191.65, 135.9]

# Set position of bar on X axis
br1 = np.arange(len(ds_800k))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
 
#make plot
plt.bar(br1, ds_100k, color = color1, width = barWidth,
        edgecolor ='grey', label ='100K')
plt.bar(br2, ds_200k, color = color2, width = barWidth,
        edgecolor ='grey', label ='200K')
plt.bar(br3, ds_400k, color = color3, width = barWidth,
        edgecolor ='grey', label ='400K')
plt.bar(br4, ds_800k, color = color4, width = barWidth,
        edgecolor ='grey', label ='800K')
plt.bar(br5, ds_1000k, color = color5, width = barWidth,
        edgecolor ='grey', label ='1M')
 
# Adding Xticks
plt.xlabel('Dataset', fontsize = 10)
plt.ylabel('Speedup', fontsize = 10)
plt.xticks([r + barWidth for r in range(len(ds_800k))],
        ['3DRoad','Porto', '3DIono', 'KITTI'])
 
plt.legend()
plt.savefig("Plots/DataSize_sqrt.pdf")'''


'''#Impact of k
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize =(8, 6))
 
# 99th percentile
ds_100k = [2.32, 23.8, 11.4, 14.5]
ds_200k = [2.6, 33.7, 11.5, 18.3]
ds_400k = [2.43, 47.1, 13.9, 21]
ds_800k = [0, 61.1, 20, 22.6]


# Set position of bar on X axis
br1 = np.arange(len(ds_800k))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

 
#make plot
plt.bar(br1, ds_100k, color = color1, width = barWidth,
        edgecolor ='grey', label ='100K')
plt.bar(br2, ds_200k, color = color2, width = barWidth,
        edgecolor ='grey', label ='200K')
plt.bar(br3, ds_400k, color = color3, width = barWidth,
        edgecolor ='grey', label ='400K')
plt.bar(br4, ds_800k, color = color4, width = barWidth,
        edgecolor ='grey', label ='800K')

 
# Adding Xticks
plt.xlabel('Dataset', fontsize = 10)
plt.ylabel('Speedup', fontsize = 10)
plt.xticks([r + barWidth for r in range(len(ds_800k))],
        ['3DRoad','Porto', '3DIono', 'KITTI'])
 
plt.legend()
plt.savefig("Plots/99thPercentile.pdf")'''




'''#Raw exec time
import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.15
fig = plt.subplots(figsize =(8, 6))
 
# 99th percentile
ds_100k = [23.8, 11.4, 14.5]
ds_200k = [33.7, 11.5, 18.3]
ds_400k = [47.1, 13.9, 21]
ds_800k = [61.1, 20, 22.6]

# DS size
ds_100k = [7.12, 28.6, 32.5, 37.7]
ds_200k = [6.13, 44.2, 54.6, 53.7]
ds_400k = [6.26, 99.9, 76.4, 71.3]
ds_800k = [0, 65.1, 73.6, 209.7]

#DS size, k=sqrt -- raw execution time per dataset
#3DIono
ds_100k = [9.48, 308.16]
ds_200k = [38.16, 2086.13]
ds_400k = [149.6, 11433.5]
ds_800k = [776.96, 57184.7]
ds_1000K = [973.3, 186528]

 
# Set position of bar on X axis
br1 = np.arange(len(ds_800k))
br2 = [x + barWidth for x in br1]

 
# Make the plot
plt.bar(br1, ds_100k, color ='r', width = barWidth,
        edgecolor ='grey', label ='TrueKNN')
plt.bar(br2, ds_200k, color ='g', width = barWidth,
        edgecolor ='grey', label ='Baseline')

# Adding Xticks
plt.xlabel('Dataset', fontsize = 10)
plt.ylabel('Execution time', fontsize = 10)
plt.xticks([r + barWidth for r in range(len(ds_800k))],['100K','200K', '400K', '800K','1M' ])
        #['3DRoad','Porto', '3DIono', 'KITTI'])
 
plt.legend()
plt.savefig("Plots/3dIono_99th_k=5")'''