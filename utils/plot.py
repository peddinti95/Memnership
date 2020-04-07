import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    #print(cm)
    #plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.show()

def drawLossAcc(plot_result,plot_title,path):
    """accuracy and loss train/value graph"""
    plt.figure()
    #plt.subplot(2,2,1)
    #plt.cla()
    #plt.plot(plot_result[0], color='#1a53ff')
    #plt.ylabel('loss train')
    #plt.subplot(2,2,2)
    #plt.plot(plot_result[1], color='#1a53ff')
    #plt.ylabel('accuracy train')
    #plt.subplot(2,2,3)
    #plt.plot(plot_result[2], color='#1a53ff')
    #plt.ylabel('loss val')
    #plt.xlabel('Epoch')
    plt.subplot(2, 2, 4)
    plt.plot(plot_result[3], color='#1a53ff')
    plt.ylabel('accuracy val')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plot_path = path + "/fig_" + plot_title + "_target_model.png"
    plt.savefig(plot_path)

def drawPlot(accuracy_per_class,precision_per_class,recall_per_class,plot_title,path):
    """my code starts to draw plot graph"""
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.hist(accuracy_per_class,bins=10,color='#1a53ff')
    plt.title("Results on "+plot_title)
    plt.ylabel('Accuracy')
    plt.subplot(3, 1, 2)
    plt.hist(recall_per_class,bins=10,color='#1a53ff')
    plt.ylabel('Recall')
    plt.subplot(3, 1, 3)
    plt.hist(precision_per_class,bins=10,color='#1a53ff')
    plt.ylabel('Precision')
    plt.tight_layout()
    plot_path=path+"/fig_"+plot_title+".png"
    plt.savefig(plot_path)

def plotterFunction(plot_title, path):
    """my code starts to draw plot graph"""
    data = pd.DataFrame()
    f1 = np.load(path + '/res_accuracy_per_class.npy') * 100
    f2 = np.load(path + '/res_precision_per_class.npy') * 100
    f3 = np.load(path + '/res_recall_per_class.npy') * 100
    data['Classlabel'] = [i + 1 for i in range(len(f1.tolist()[0]))]
    data['Accuracy'] = f1.tolist()[0]
    data['Precision'] = f2.tolist()[0]
    data['Recall'] = f3.tolist()[0]
    # Plot the figure for Accuracy vs Class label
    plt.figure(figsize=(10,8))
    ax1 = sns.barplot(x="Classlabel", y="Accuracy", data=data, palette="Blues_d")
    for index, row in data.iterrows():
        ax1.text(row.Classlabel - 1, row.Accuracy, round(row.Accuracy, 2), color='black', ha="center")
    plt.title("Accuracy vs Class label on " + plot_title)
    plot_acc_save = path + "/fig_accuracy" + plot_title + ".png"
    plt.savefig(plot_acc_save)
    # Plot the figure for Precision vs Class label
    plt.figure(figsize=(10,8))
    ax2 = sns.barplot(x="Classlabel", y="Precision", data=data, palette="Blues_d")
    for index, row in data.iterrows():
        ax2.text(row.Classlabel - 1, row.Precision, round(row.Precision, 2), color='black', ha="center")
    plt.title("Precision vs Class label on " + plot_title)
    plot_pre_save = path + "/fig_precision" + plot_title + ".png"
    plt.savefig(plot_pre_save)
    # Plot the figure for Recall vs Class label
    plt.figure(figsize=(10,8))
    ax3 = sns.barplot(x="Classlabel", y="Recall", data=data, palette="Blues_d")
    for index, row in data.iterrows():
        ax3.text(row.Classlabel - 1, row.Recall, round(row.Recall, 2), color='black', ha="center")
    plt.title("Recall vs Class label on " + plot_title)
    plot_rec_save = path + "/fig_recall" + plot_title + ".png"
    plt.savefig(plot_rec_save)
