import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    TEXT_PATH = r'D:/GoogleDrive/kysely/'  # data folder

    data = pickle.load(open(TEXT_PATH + r'SURPRISE_RESULTS.pickle', 'rb'))

    data_matrices = pickle.load(open(TEXT_PATH+'Laurea_limesurvey_experiment_16.01.2018_RESPONSE_MATRIX.pickle','rb'))

    dimensions = ['reliability','sentiment','infovalue','subjectivity','textlogic','writestyle']
    ratings = []
    fignum=0

    def sorter(data,ind):
        return [data[i] for i in ind]

    for dimension in dimensions:

        simple_mean = data[dimension]['item_mean']
        baseline_als_bias = data[dimension]['BASELINE_ALS']['item_bias']
        baseline_sgd_bias = data[dimension]['BASELINE_SGD']['item_bias']
        nmf_bias = data[dimension]['NMF']['item_bias']

        x_ind = np.array(range(len(baseline_als_bias)))

        x_label1 = list(baseline_als_bias.keys())
        x_label2 = list(baseline_sgd_bias.keys())
        x_label3 = list(nmf_bias.keys())
        x_label4 = list(simple_mean.keys())

        legends = ('baseline ALS', 'baseline SGD', 'NMF', 'simple mean')

        x_data1 = list(baseline_als_bias.values())
        x_data2 = list(baseline_sgd_bias.values())
        x_data3 = list(nmf_bias.values())
        x_data4 = np.array(list(simple_mean.values()))

        x_data4 = x_data4 - np.mean(x_data4)

        ratings.append(x_data2.copy())

        ind = np.argsort(x_data2)
        x_label = [x_label1[i] for i in ind]

        x_data1 = sorter(x_data1,ind)
        x_data2 = sorter(x_data2,ind)
        x_data3 = sorter(x_data3,ind)
        x_data4 = sorter(x_data4,ind)

        assert x_label1==x_label2==x_label3==x_label4,'Items not equal!'

        fignum+=1
        #width = 0.1  # the width of the bars

        fig = plt.figure(num=fignum)
        DPI = float(fig.dpi)
        fig.set_size_inches(1800/DPI, 800/DPI)

        width=0.2
        rects1 = plt.bar(x_ind,x_data1,width,color='r')
        rects2 = plt.bar(x_ind + width, x_data2, width, color='b')
        rects3 = plt.bar(x_ind + 2 * width, x_data3, width, color='y')
        rects4 = plt.bar(x_ind + 3 * width, x_data4, width, color='g')

        # add some text for labels, title and axes ticks
        ax = fig.get_axes()[0]
        ax.set_ylabel('Rating value')
        ax.set_xlabel('Text ID')
        ax.set_title('Ratings for \'%s\'' % dimension)
        ax.set_xticks(x_ind)
        ax.set_xticklabels(x_label,rotation=-60)
        ax.tick_params(axis='x', which='both', labelsize=6)
        ax.tick_params(axis='y',labelsize=18)

        ax.set_xlim([x_ind[0]-3,x_ind[-1]+3])
        #ax.autoscale(enable=True, axis='x', tight=True)


        plt.legend((rects1[0], rects2[0],rects3[0],rects4[0]),legends)

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%0.1d' % (height),
                        ha='center', va='bottom')

        #autolabel(rects1)
        #autolabel(rects2)
        #autolabel(rects3)
        #autolabel(rects4)

        plt.tight_layout()

        mat = [x_data1,x_data2,x_data3,x_data4]
        mat = np.corrcoef(np.array(mat))
        df = pandas.DataFrame(mat)

        # this is an inset axes over the main axes
        a = plt.axes([.70,.22,.25,.25], facecolor='w')
        #a.set_aspect('equal')
        #plt.plot([0,2,4,6,7],[9,1,4,9,0])
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        sns.set_style(style = 'white')
        # Add diverging colormap from red to blue
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        # Draw correlation plot with or without duplicates
        sns.heatmap(df, mask=mask,square=True,linewidth=.5, cbar_kws={"shrink": .5}, ax=a,xticklabels=legends,yticklabels=legends,annot=True, fmt="0.2f")

        a.set_yticklabels(legends, rotation=0)
        a.set_xticklabels(legends, rotation=-40)

        plt.show()


    fignum+=1

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    #plt.tight_layout()

    DPI = float(fig.dpi)
    fig.set_size_inches(1300/DPI,500/DPI)

    mat = np.corrcoef(np.array(ratings))
    df = pandas.DataFrame(mat)

    # this is an inset axes over the main axes

    # a.set_aspect('equal')
    # plt.plot([0,2,4,6,7],[9,1,4,9,0])
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.set_style(style='white')
    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    # Draw correlation plot with or without duplicates
    sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5},ax=ax1,xticklabels=dimensions,yticklabels=dimensions,annot=True, fmt="0.2f")

    ax1.set_yticklabels(dimensions, rotation=0)
    ax1.set_xticklabels(dimensions, rotation=-40)
    ax1.set_title('Rating SGD')


    mat = np.zeros((len(dimensions),len(dimensions)))
    count_mat = np.zeros((len(dimensions), len(dimensions)))
    subjects=data_matrices[0][dimensions[0]].columns
    N=len(subjects)
    for col in subjects:
        ind = data_matrices[0][dimensions[0]][col].notnull()
        for k1,dim1 in enumerate(dimensions):
            for k2,dim2 in enumerate(dimensions):
                val1 = np.array(data_matrices[0][dim1].loc[ind, col])
                val2 = np.array(data_matrices[0][dim2].loc[ind, col])
                r = np.corrcoef(val1,val2)[0,1]
                if not(np.isnan(r)):
                    mat[k1,k2] += r
                    count_mat[k1,k2]+=1
    mat = mat/count_mat

    df = pandas.DataFrame(mat)

    # this is an inset axes over the main axes

    # a.set_aspect('equal')
    # plt.plot([0,2,4,6,7],[9,1,4,9,0])
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.set_style(style='white')
    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    # Draw correlation plot with or without duplicates
    sns.heatmap(df, mask=mask, cmap=cmap, square=True, linewidth=.5, cbar_kws={"shrink": .5},ax=ax2,xticklabels=dimensions,yticklabels=dimensions,annot=True, fmt="0.2f")

    ax2.set_yticklabels(dimensions, rotation=0)
    ax2.set_xticklabels(dimensions, rotation=-40)
    ax2.set_title('Raw ratings (mean over subjects)')

    plt.subplots_adjust(left  = 0.125,  # the left side of the subplots of the figure
                        right = 0.9,    # the right side of the subplots of the figure
                        bottom = 0.2,   # the bottom of the subplots of the figure
                        top = 0.9,      # the top of the subplots of the figure
                        wspace = 0.0,   # the amount of width reserved for blank space between subplots
                        hspace = 0.0)   # the amount of height reserved for white space between subplots

    plt.show()