import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

color1 = (148/255.0,  53/255.0,  65/255.0)
color2 = (80/255.0, 132/255.0, 167/255.0)
color3 = (81/255.0, 196/255.0, 112/255.0)
dark_green = (0/255.0, 100/255.0, 0/255.0)
dark_blue = (0/255.0, 0/255.0, 139/255.0)

# plot_alu('2dspeedup', 8, 7, 0.25, 5.0, 2.5)
def plot_alu(filename, nreal, width, w, h, rotation=30, text_placement=0, decimals = '%.1f'):
    df = pd.read_csv(filename+'.csv', sep = ',')
    print(df)
    # return

    benchmarks = list(df['Dataset_Size'][:])
    n = nreal

    fig, ax = plt.subplots(figsize=(20,20))
    # ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    index = np.arange(n)
    print(index)
    plt.bar(index + -1 * width, list(df['RTDBSCAN'][:]), width, label='RT-DBSCAN', color=color1, alpha=.99, hatch="////") 
    
    '''max_list = list(map(min, zip(list(df['FDBSCAN'][:]), [2.5] * n)))
    plt.bar(index +  0 * width, max_list, width, label='FDBSCAN', color=color2) 
    for i, v in enumerate([2.5] * n):
        if (df['FDBSCAN'][i] > 2.5) :
            plt.text(i + -2.35 * width, 2.32, decimals % df['FDBSCAN'][i], fontsize='small', fontweight='bold',color=dark_blue)'''
    
  

    plt.axhline(y=1.0, color='black', linestyle='dotted') 
    # plt.axhline(y=0.5, color='white', linestyle='solid', linewidth=0.1)
    # plt.axhline(y=1.5, color='white', linestyle='solid', linewidth=0.1) 
    # plt.axhline(y=2.0, color='white', linestyle='solid', linewidth=0.1) 
    
    plt.ylim(0, 2.55)
    plt.xlim(-0.75, n-0.25)
    
    plt.ylabel('Normalized Execution Time', fontsize='large')
    plt.xticks(index, benchmarks, fontsize='small', rotation=rotation)
    plt.yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.1], fontsize='medium')
    plt.setp(ax.get_yticklabels()[-1], visible=False)
    plt.legend(loc='upper center', ncol=3, fontsize='medium') 
    # plt.legend(bbox_to_anchor=(0.4,0.8), loc="upper right")
    # plt.legend(loc='upper center', bbox_to_anchor=(0, 1.02, 1, 1.02), ncol=3, fontsize='small')
    # plt.legend(bbx_to_anchor)

    print('hih hih')
        
    plt.tight_layout()
    plt.gcf().set_size_inches(w,h)
    plt.gcf().savefig(filename+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.clf()

if __name__ == "__main__":
    plot_alu('3diono_results', 4, 0.25, 4.0, 2.5)
