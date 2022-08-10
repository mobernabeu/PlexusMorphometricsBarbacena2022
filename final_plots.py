from plexus_analysis import SproutingFront, NetworkSkeleton
import os.path
import numpy as np
import itertools
import scipy.stats
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import colors

# Some of the code based on these examples:
# https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2
# https://realpython.com/k-means-clustering-python/#how-to-perform-k-means-clustering-in-python

# Make cluster ids reproducible
np.random.seed(seed=2022)

feature_names = ['label', 'vessel density', 'bifurcation density', 'avascular spaces density', 'range of avascular area','mean avascular area',
                 'std of avascular area', 'mean diameter', 'std of diameter']

bin_width = 100.0
ranges = np.arange(0., 901., bin_width)

regions_cmap = colors.ListedColormap(['aliceblue', 'mistyrose']) # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
regions_cmap = colors.ListedColormap(['lightskyblue', 'lightsalmon']) # https://matplotlib.org/3.1.0/gallery/color/named_colors.html

def calculateFeatures(group_name='PBS'):
    from dataset_list import datasets

    assert group_name in datasets.keys()

    group_values = []
    for images_id, images_name in enumerate(datasets[group_name]):
        retina_values = []
        for image_name in images_name:
            a_dir = os.path.join(group_name, image_name)
            print image_name

            sp = SproutingFront(os.path.join(a_dir, 'config.yml'))
            skeleton_filename = os.path.join(a_dir, image_name + '.vtp')
            skeleton = NetworkSkeleton(skeleton_filename, sp)

            features = []

            # feature 0 (labels), use top of the range as label, i.e. [0, 100] is labelled 100
            features.append(ranges[1:])

            # feature 1
            vascularised_densities = np.array([skeleton.vascularised_density_for_range((ranges[i], ranges[i+1])) for i in range(0, len(ranges)-1)])
            features.append(vascularised_densities)
            
            # feature 2
            bifurcation_densities = np.array([skeleton.bifurcation_density_for_range((ranges[i], ranges[i+1])) for i in range(0, len(ranges)-1)])
            features.append(bifurcation_densities)

            # feature 3
            faces = pickle.load(open(os.path.join(a_dir,'faces.obj'), 'rb'))
            areas_no_filter = np.array([face.area() for face in faces])
            # Filter out face which corresponds to the whole network
            assert(np.sum(areas_no_filter > 1e5) <= 1)
            areas = areas_no_filter[areas_no_filter < 1e5] #/ area_normalisation
            faces = itertools.compress(faces, areas_no_filter < 1e5)
            dist_sp = [sp.distance(face.barycenter()) for face in faces]
            face_count, _, _ = scipy.stats.binned_statistic(dist_sp, areas, statistic='count', bins=ranges)
            bin_areas = [skeleton.area_for_range((ranges[i], ranges[i+1])) for i in range(0, len(ranges)-1)]
            face_densities = face_count/bin_areas
            features.append(face_densities)

            # feature 4
            face_area_ranges, _, _ = scipy.stats.binned_statistic(dist_sp, areas, statistic=np.ptp, bins=ranges)
            features.append(face_area_ranges)

            # feature 5
            face_area_means, _, _ = scipy.stats.binned_statistic(dist_sp, areas, statistic='mean', bins=ranges)
            features.append(face_area_means)

            # feature 6
            face_area_std, _, _ = scipy.stats.binned_statistic(dist_sp, areas, statistic='std', bins=ranges)
            features.append(face_area_std)

            # feature 7
            diameters = [skeleton.diameters_for_range((ranges[i], ranges[i+1])) for i in range(0, len(ranges)-1)]
            diameter_means = np.array([np.mean(diameters_for_bin) for diameters_for_bin in diameters])
            features.append(diameter_means)

            # feature 8
            diameter_std = np.array([np.std(diameters_for_bin) for diameters_for_bin in diameters])
            features.append(diameter_std)

            # feature 9: retina id
            features.append([images_id] * len(ranges[1:]))

            retina_values.append(features)
            
        group_values.append(np.mean(np.array(retina_values), axis=0))

    group_values = np.array(group_values)

    # It should be possible to do this with .reshape
    num_image, num_features, num_bins = group_values.shape
    print num_image, num_features, num_bins
    data_points = []
    for image_id in range(num_image):
        for bin_id in range(num_bins):
            data_points.append(group_values[image_id, :, bin_id])

    data_points = np.array(data_points)
    df = pd.DataFrame(data=data_points)

    df.to_csv('bin_features_{}.csv'.format(group_name), index=False)

def plot_mean_features_per_bin(data):

    bins = sorted(set(df['0']))

    for feature_id, feature_name in enumerate(data.columns):
        feature_means = []
        feature_sems = []
        for bin_id in bins:
            mean = np.mean(data[feature_name].loc[df['0'] == bin_id])
            feature_means.append(mean)
            sem = scipy.stats.sem(data[feature_name].loc[df['0'] == bin_id])
            feature_sems.append(sem)
        #plt.plot(bins, feature_means)
        plt.errorbar(bins, feature_means, feature_sems)     
        name = feature_names[feature_id]
        plt.title(name)
        plt.savefig('{}.{}'.format(name, file_format))
        plt.clf()

def plot_decision_boundaries(classifier, region_of_interest):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = region_of_interest[:, 0].min() - 1, region_of_interest[:, 0].max() + 1
    y_min, y_max = region_of_interest[:, 1].min() - 1, region_of_interest[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=regions_cmap,
               aspect='auto', origin='lower')


def plot_histograms(group_name, labels, bin_ids):
    unique_bin_id = np.unique(bin_ids)

    label_counts = []
    for cluster_id in range(2):
        ix = np.where(labels == cluster_id)
        distances_in_cluster = bin_ids[ix]
        label_counts.append([np.count_nonzero(distances_in_cluster == g) for g in unique_bin_id])

    width = 20
    shifts = [0.5*width, -0.5*width]
    for cluster_id in range(2):
        shift = shifts[cluster_id]
        plt.bar(unique_bin_id-shift, label_counts[cluster_id], width, label='class {}'.format(cluster_id), color=regions_cmap(cluster_id))

    plt.xticks(np.arange(min(unique_bin_id), max(unique_bin_id)+1, 100))
    plt.ylabel('Count')
    plt.xlabel('Maximum distance from sprouting front')
    plt.legend()
    plt.savefig('{}_histogram.{}'.format(group_name, file_format))
    plt.clf()


if __name__ == '__main__':

    file_format = 'eps'

    def read_csv(file_name, columns=[0,3,4]):       
        df = pd.read_csv(file_name)
        df = df.dropna()

        #plot_mean_features_per_bin(df)
        #exit(-1)

        return drop_rows(drop_columns(df))

    def drop_columns(df):
        drop_columns = ['3', '4']
        drop_columns = ['3']
        
        new_df = df.drop(columns=drop_columns)

        return new_df

    def drop_rows(df):
        # drop_rows = [700, 800, 900]
        drop_rows = []        

        new_df = df[~df['0'].isin(drop_rows)]

        return new_df


    # calculateFeatures('PBS')
    df_pbs = read_csv('bin_features_PBS.csv')
    #df_pbs['group'] = 'PBS'

    # calculateFeatures('ANGIOTENSIN')
    df_angio = read_csv('bin_features_ANGIOTENSIN.csv')
    #df_angio['group'] = 'ANGIOTENSIN'

    # calculateFeatures('SFLT1')
    df_sflt1 = read_csv('bin_features_SFLT1.csv')
    #df_sflt1['group'] = 'SFLT1'

    # calculateFeatures('VEGF')
    df_vegf = read_csv('bin_features_VEGF.csv')
    #df_vegf['group'] = 'VEGF'

    # calculateFeatures('CAPTOPRIL')
    df_capto = read_csv('bin_features_CAPTOPRIL.csv')
    #df_capto['group'] = 'CAPTOPRIL'


    # Standardize the data to have a mean of ~0 and a variance of 1
    exclude_column_from_pca = ['0', '9']
    scaler = StandardScaler().fit(df_pbs.drop(exclude_column_from_pca, axis=1)) 
    pbs_std = scaler.transform(df_pbs.drop(exclude_column_from_pca, axis=1))
    angio_std = scaler.transform(df_angio.drop(exclude_column_from_pca, axis=1))
    sflt1_std = scaler.transform(df_sflt1.drop(exclude_column_from_pca, axis=1))
    vegf_std = scaler.transform(df_vegf.drop(exclude_column_from_pca, axis=1))
    capto_std = scaler.transform(df_capto.drop(exclude_column_from_pca, axis=1))

    # Calculate PCA on PBS group and project other groups on same plane
    pca = PCA(n_components=2)

    pbs_pca = pca.fit_transform(pbs_std)
    print 'PCA0', pca.components_[0]
    print 'PCA1', pca.components_[1]

    # Alternatively, do PCA using all the groups
    # pca.fit(np.concatenate([pbs_std, angio_std, sflt1_std, vegf_std, capto_std]))
    # pbs_pca = pca.transform(pbs_std)
    
    angio_pca = pca.transform(angio_std)
    sflt1_pca = pca.transform(sflt1_std)
    vegf_pca = pca.transform(vegf_std)
    capto_pca = pca.transform(capto_std)

    # Cluster PBS data
    kmeans = KMeans(n_clusters=2, n_init=25)
    kmeans.fit(pbs_pca)

    plot_decision_boundaries(kmeans, pbs_pca)

    group = df_pbs[df_pbs.columns[0]].to_numpy()
    unique_group = np.unique(group)
    colors = cm.rainbow(np.linspace(0, 1, len(unique_group)))
    for g, col in zip(unique_group, colors):
        ix = np.where(group == g)
        plt.scatter(pbs_pca[ix, 0], pbs_pca[ix, 1], color=col, label=int(g))
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # plt.ylim(-5, 6)
    plt.legend()
    plt.savefig('pca_pbs.{}'.format(file_format))

    plt.clf()


    plot_decision_boundaries(kmeans, np.concatenate([pbs_pca, sflt1_pca, vegf_pca, capto_pca, angio_pca]))

    plt.scatter(pbs_pca[:, 0], pbs_pca[:, 1], label='pbs', color='0.7', s=9) # From a colour greyscale [0, 1]
    plt.scatter(angio_pca[:, 0], angio_pca[:, 1], label='angio', color='red', s=9)
    plt.scatter(sflt1_pca[:, 0], sflt1_pca[:, 1], label='sflt1', color='salmon', s=9)
    plt.scatter(vegf_pca[:, 0], vegf_pca[:, 1], label='vegf', color='blue', s=9)
    plt.scatter(capto_pca[:, 0], capto_pca[:, 1], label='capto', color='deepskyblue', s=9)

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.legend()
    plt.savefig('pca_all_groups.{}'.format(file_format))

    plt.clf()


    pbs_labels = kmeans.labels_
    kneigh = KNeighborsClassifier(n_neighbors=3)
    kneigh.fit(pbs_pca, pbs_labels)

    angio_labels = kneigh.predict(angio_pca)
    sflt1_labels = kneigh.predict(sflt1_pca)
    vegf_labels = kneigh.predict(vegf_pca)
    capto_labels = kneigh.predict(capto_pca)

    def plot_class_ratios():
        ratios = [np.sum(pbs_labels == 0)/float(len(pbs_labels)),
                  np.sum(angio_labels == 0)/float(len(angio_labels)),
                  np.sum(sflt1_labels == 0)/float(len(sflt1_labels)),
                  np.sum(vegf_labels == 0)/float(len(vegf_labels)),
                  np.sum(capto_labels == 0)/float(len(capto_labels))]

        plt.bar(range(len(ratios)), ratios, color=regions_cmap(0))
        plt.xticks(range(5), ['PBS', 'Angiotensin', 'SFLT1', 'VEGF', 'Captopril'])

        plt.xlabel('Group')
        plt.ylabel('Class 0 ratio')
        plt.savefig('class0_ratio_all_groups.{}'.format(file_format))
        plt.clf()
    
    # plot_class_ratios()


    def plot_class_ratios_per_retina(class_id):
        other_class_id = 0 if class_id==1 else 1

        data = [[df_pbs, pbs_labels],
                [df_angio, angio_labels],
                [df_sflt1, sflt1_labels],
                [df_vegf, vegf_labels],
                [df_capto, capto_labels]]

        data_per_group = []
        means = []
        sems = []

        for df, labels in data:
            df['label'] = labels
            retina_ids = np.unique(df['9'])
            ratios = []
            for retina_id in retina_ids:
                this_retina = df.loc[df['9'] == retina_id]
                ratio = np.sum(this_retina['label'] == class_id)/float(len(this_retina))
                # ratio = np.sum(this_retina['label'] == class_id)
                # ratio = float(len(this_retina))
                # ratio = np.sum(this_retina['label'] == class_id)/float(np.sum(this_retina['label'] == other_class_id))
                ratios.append(ratio)
            data_per_group.append(ratios)
            means.append(np.mean(ratios))
            sems.append(scipy.stats.sem(ratios))

        tests = []
        for datum in data_per_group:
            print datum
            # stat, p = scipy.stats.mannwhitneyu(data_per_group[0], datum)
            stat, p = scipy.stats.mannwhitneyu(data_per_group[0], datum, alternative='two-sided')            
            #stat, p = scipy.stats.kruskal(data_per_group[0], datum)
            #stat, p = scipy.stats.ttest_ind(data_per_group[0], datum, equal_var=False)
            #stat, p = scipy.stats.f_oneway(data_per_group[0], datum)
            tests.append(p)#(p<0.05)

        print 'p-values', tests        

#        import pdb; pdb.set_trace()

        plt.bar(range(len(means)), means, yerr=sems, color=regions_cmap(class_id))
        plt.xticks(range(5), ['PBS', 'Angiotensin', 'SFLT1', 'VEGF', 'Captopril'])

        plt.xlabel('Group')
        # plt.ylabel('Class 0 ratio')
        plt.ylabel('Class {} ratio'.format(class_id))
        plt.savefig('class{}_ratio_per_retina_all_groups.{}'.format(class_id, file_format))
        plt.clf()
    
    print 'Class 0'
    plot_class_ratios_per_retina(0)
    print 'Class 1'
    plot_class_ratios_per_retina(1)

    plot_histograms('PBS', pbs_labels, df_pbs['0'].to_numpy())
    plot_histograms('Angiotensin', angio_labels, df_angio['0'].to_numpy())
    plot_histograms('SFLT1', sflt1_labels, df_sflt1['0'].to_numpy())
    plot_histograms('VEGF', vegf_labels, df_vegf['0'].to_numpy())
    plot_histograms('Captopril', capto_labels, df_capto['0'].to_numpy())
