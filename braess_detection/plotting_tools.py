def _gen_signatures(Gr, dists, maxflow_edge, be, notbe, noeff, aligned, notaligned, cant_say):
    dists_array = []
    braess_signature_array = []
    aligned_signature_array = []

    edge_dist = lambda e1,e2:min(dists[e1[0]][e2[0]], dists[e1[0]][e2[1]], dists[e1[1]][e2[0]], dists[e1[1]][e2[1]])
    for e in Gr.edges_arr:
        if e == maxflow_edge:
            continue
        d = edge_dist(maxflow_edge,e)
        dists_array.append(d)

        if e in be:
            braess_signature_array.append(1)
        elif e in notbe:
            braess_signature_array.append(0)
        else:
            assert(e in noeff)
            braess_signature_array.append(-1)

        if e in aligned:
            aligned_signature_array.append(1)
        elif e in notaligned:
            aligned_signature_array.append(0)
        else:
            assert(e in cant_say)
            aligned_signature_array.append(-1)

    return dists_array, braess_signature_array, aligned_signature_array

def bla():
    dists_array, braess_signature_array, aligned_signature_array\
            = _gen_signatures(Gr, dists, maxflow_edge, be, notbe, noeff, aligned,\
                             notaligned, cant_say)
    return np.vstack((dists_array, braess_signature_array, aligned_signature_array)).T


def evaluate_edist_alignment_classifier(G, Gr, I, dists=None, thres = 0.00005):
    if dists is None:
        dists = dict(nx.all_pairs_shortest_path_length(G))

    stflows_dict, maxflow_edge = _stflow_and_maxflow_edge(Gr, I)
    be, notbe, noeff = braessian_edges(G, I, thres = thres, Gr = Gr)

    aligned, notaligned, cant_say = alignment_flows(G, maxflow_edge,
                                                    stflows_dict, dists=dists)
    dists_array, braess_signature_array, aligned_signature_array\
            = _gen_signatures(Gr, dists, maxflow_edge, be, notbe, noeff, aligned,\
                             notaligned, cant_say)
    return np.vstack((dists_array, braess_signature_array, aligned_signature_array)).T


def goodness_overall(data, normed=False, axis=None, title=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots(figsize=(10,6))
    d1 = data
    be_al = (d1[:,[1,2]]==[1,1]).all(axis=1).sum()
    notbe_al = (d1[:,[1,2]]==[0,1]).all(axis=1).sum()
    noeff_al = (d1[:,[1,2]]==[-1,1]).all(axis=1).sum()

    be_notal = (d1[:,[1,2]]==[1,0]).all(axis=1).sum()
    notbe_notal = (d1[:,[1,2]]==[0,0]).all(axis=1).sum()
    noeff_notal = (d1[:,[1,2]]==[-1,0]).all(axis=1).sum()

    be_cantsay = (d1[:,[1,2]]==[1,-1]).all(axis=1).sum()
    notbe_cantsay = (d1[:,[1,2]]==[0,-1]).all(axis=1).sum()
    noeff_cantsay = (d1[:,[1,2]]==[-1,-1]).all(axis=1).sum()

    df = pd.DataFrame(data={'aligned': [be_al, notbe_al, noeff_al],
                            'anti-aligned': [be_notal, notbe_notal, noeff_notal],
                            'undefined': [be_cantsay, notbe_cantsay, noeff_cantsay]
                           },
                      index=['BE', 'not BE', 'no effect'])
    #ax.axis('off')
    if normed==True:
        df /= df.sum().sum()
        fmt = '0.4f'
    else:
        fmt = 'd'
    sns.heatmap(df, annot=True, cbar=False, fmt=fmt, ax=axis, **kwargs)

    for tick in axis.get_xticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(40)
    for tick in axis.get_yticklabels():
        tick.set_rotation(45)
        tick.set_fontsize(40)

    if title is not None:
        axis.set_title(title, fontsize=40, y=1.1)

def goodness_two_distances(data):
    fig, axes = plt.subplots(1,2, figsize=(20,6))
    for idx,d in enumerate([0,1]):
        d1 = data[data[:,0]==d,:]
        be_al = (d1[:,[1,2]]==[1,1]).all(axis=1).sum()
        notbe_al = (d1[:,[1,2]]==[0,1]).all(axis=1).sum()
        noeff_al = (d1[:,[1,2]]==[-1,1]).all(axis=1).sum()

        be_notal = (d1[:,[1,2]]==[1,0]).all(axis=1).sum()
        notbe_notal = (d1[:,[1,2]]==[0,0]).all(axis=1).sum()
        noeff_notal = (d1[:,[1,2]]==[-1,0]).all(axis=1).sum()

        be_cantsay = (d1[:,[1,2]]==[1,-1]).all(axis=1).sum()
        notbe_cantsay = (d1[:,[1,2]]==[0,-1]).all(axis=1).sum()
        noeff_cantsay = (d1[:,[1,2]]==[-1,-1]).all(axis=1).sum()

        df = pd.DataFrame(data={'aligned': [be_al, notbe_al, noeff_al],
                                'not aligned': [be_notal, notbe_notal, noeff_notal],
                                'undefined': [be_cantsay, notbe_cantsay, noeff_cantsay]
                               },
                          index=['BE', 'not BE', 'no effect'])

    #ax.axis('off')

        sns.heatmap(df, annot=True, cbar=False, fmt='d', ax=axes[idx])
        axes[idx].set_title("Distance=%d"%d, fontsize = 30)


def barchart_two_distances_w_errorbar(data):
    df = pd.DataFrame(data, columns = ('dist', 'BE', 'aligned', 'rep_idx'), dtype=int)
    df.loc[:, 'BE, aligned'] = (df['BE']==1) & (df['aligned']==1)
    df.loc[:, 'not BE, aligned'] = (df['BE']==0) & (df['aligned']==1)
    df.loc[:, 'no effect, aligned'] = (df['BE']==-1) & (df['aligned']==1)

    df.loc[:, 'BE, anti-aligned'] = (df['BE']==1) & (df['aligned']==0)
    df.loc[:, 'not BE, anti-aligned'] = (df['BE']==0) & (df['aligned']==0)
    df.loc[:, 'no effect, anti-aligned'] = (df['BE']==-1) & (df['aligned']==0)

    df.loc[:, 'BE, undefined'] = (df['BE']==1) & (df['aligned']==-1)
    df.loc[:, 'not BE, undefined'] = (df['BE']==0) & (df['aligned']==-1)
    df.loc[:, 'no effect, undefined'] = (df['BE']==-1) & (df['aligned']==-1)

    # first dists 1 and 2
    gr = (df.groupby(['dist','rep_idx']).agg('sum')/df.groupby(['dist','rep_idx']).agg('count')).reset_index()
    wide_df = gr[['rep_idx', 'dist', 'BE, aligned', 'not BE, anti-aligned', 'BE, anti-aligned', 'not BE, aligned', 'no effect, undefined', 'BE, undefined', 'not BE, undefined',\
                            'no effect, aligned', 'no effect, anti-aligned']]
    long_df = pd.melt(wide_df, id_vars = ['rep_idx', 'dist'], value_vars=['BE, aligned', 'not BE, anti-aligned', 'BE, anti-aligned', 'not BE, aligned', 'no effect, undefined', 'BE, undefined', 'not BE, undefined',\
                            'no effect, aligned', 'no effect, anti-aligned'], value_name = 'fraction of edges')
    long_df_dists_2 = long_df[long_df['dist']<2]

    # now all dists
    gr = (df.groupby('rep_idx').agg('sum')/df.groupby('rep_idx').agg('count')).reset_index()
    wide_df = gr[['rep_idx', 'BE, aligned', 'not BE, anti-aligned', 'BE, anti-aligned', 'not BE, aligned', 'no effect, undefined', 'BE, undefined', 'not BE, undefined',\
                            'no effect, aligned', 'no effect, anti-aligned']]
    long_df = pd.melt(wide_df, id_vars = ['rep_idx'], value_vars=['BE, aligned', 'not BE, anti-aligned', 'BE, anti-aligned', 'not BE, aligned', 'no effect, undefined', 'BE, undefined', 'not BE, undefined',\
                            'no effect, aligned', 'no effect, anti-aligned'], value_name = 'fraction of edges')
    long_df.loc[:, 'dist'] = 'all'

    all_data = pd.concat((long_df, long_df_dists_2))

    pl = sns.factorplot(x="variable", y="fraction of edges", data=all_data, kind = 'bar', hue='dist', size = 8, aspect = 2.5, ci = 'sd',\
                        hue_order = (0,1,'all'), units='rep_idx', legend_out = False)
    pl.set_xticklabels(rotation=30)

    l = pl.axes[0,0].legend_
    l.set_title('Distance')
    l.get_title().set_fontsize(40)
    for t in l.get_texts():
        t.set_fontsize('40')


    ax_orig = pl.axes[0,0]
    ax_orig.set_xticklabels(ax_orig.get_xticklabels(), rotation=45, ha='right')


    ax_orig.set_ylim(0,1)
    ax_orig.set_xlabel('')

    y,h,col = 1, 0.1, 'k'
    bbox_props = {'pad':1, 'facecolor': 'white', 'boxstyle':'round','pad':0.3}
    arrowprops = dict(arrowstyle='-',    #"-",
                                connectionstyle='arc,angleA=90, angleB=90, rad=4, armA=25, armB=25'  #"bar, fraction=0.05",
                      )

    ax = ax_orig #inx()
    #x.axis('off')

    x1,x2 = -0.497,1.503
    #ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, "correct predictions", ha='center', va='bottom', color=col, fontsize = 30, bbox=bbox_props)
    ax.annotate("",
                xy=(x1,y), xycoords='data',
                xytext=(x2, y), textcoords='data',
                arrowprops=arrowprops
                )


    x1,x2 = 1.5,3.5
    ax.text((x1+x2)*.5, y+h, "false predictions", ha='center', va='bottom', color=col, fontsize = 30, bbox=bbox_props)
    ax.annotate("",
                xy=(x1,y), xycoords='data',
                xytext=(x2, y), textcoords='data',
                arrowprops=arrowprops
                )

    x1,x2 = 3.5, 8.5
    ax.text((x1+x2)*.5, y+h, "can't say", ha='center', va='bottom', color=col, fontsize = 30, bbox=bbox_props)
    ax.annotate("",
                xy=(x1,y), xycoords='data',
                xytext=(x2, y), textcoords='data',
                arrowprops=arrowprops
                )
    sns.despine(top=False, right=False)
    return pl


def barchart_two_distances(data, alldists=True):
#    sns.set_palette('deep')

    fig, ax = plt.subplots(figsize=(20,6))
    allrects = []
    ind = np.array(range(9))
    ax2 = fig.add_axes([0,0.95,1,0.1])
    ax2.axis('off')
    maxdist = max(data[:,0])


    width = 0.3
    for idx,d in enumerate([0,1]):
        d1 = data[data[:,0]==d,:]
        be_al = (d1[:,[1,2]]==[1,1]).all(axis=1).sum()
        notbe_al = (d1[:,[1,2]]==[0,1]).all(axis=1).sum()
        noeff_al = (d1[:,[1,2]]==[-1,1]).all(axis=1).sum()

        be_notal = (d1[:,[1,2]]==[1,0]).all(axis=1).sum()
        notbe_notal = (d1[:,[1,2]]==[0,0]).all(axis=1).sum()
        noeff_notal = (d1[:,[1,2]]==[-1,0]).all(axis=1).sum()

        be_cantsay = (d1[:,[1,2]]==[1,-1]).all(axis=1).sum()
        notbe_cantsay = (d1[:,[1,2]]==[0,-1]).all(axis=1).sum()
        noeff_cantsay = (d1[:,[1,2]]==[-1,-1]).all(axis=1).sum()

        y = np.array([be_al, notbe_notal, be_notal, notbe_al, noeff_cantsay, be_cantsay, notbe_cantsay, noeff_al, noeff_notal])
        y= y/np.sum(y)
        allrects.append(ax.bar(ind+idx*width, y, width))

    # now the alldists one
    if alldists == True:
        d1 = data
        be_al = (d1[:,[1,2]]==[1,1]).all(axis=1).sum()
        notbe_al = (d1[:,[1,2]]==[0,1]).all(axis=1).sum()
        noeff_al = (d1[:,[1,2]]==[-1,1]).all(axis=1).sum()

        be_notal = (d1[:,[1,2]]==[1,0]).all(axis=1).sum()
        notbe_notal = (d1[:,[1,2]]==[0,0]).all(axis=1).sum()
        noeff_notal = (d1[:,[1,2]]==[-1,0]).all(axis=1).sum()

        be_cantsay = (d1[:,[1,2]]==[1,-1]).all(axis=1).sum()
        notbe_cantsay = (d1[:,[1,2]]==[0,-1]).all(axis=1).sum()
        noeff_cantsay = (d1[:,[1,2]]==[-1,-1]).all(axis=1).sum()

        y = np.array([be_al, notbe_notal, be_notal, notbe_al, noeff_cantsay, be_cantsay, notbe_cantsay, noeff_al, noeff_notal])
        y= y/np.sum(y)
        allrects.append(ax.bar(ind+2*width, y, width))


    #ax.text(-0.3,0.75, 'Correct predictions', fontsize=20)
    ax2.annotate('Correct predictions', xy=(0.22, 0), xycoords='data',
                    fontsize=20, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=6.8, lengthB=1', lw=1.0))

    ax2.annotate('False predictions', xy=(0.397, 0), xycoords='data',
                    fontsize=20, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=6, lengthB=1', lw=1.0))

    ax2.annotate('Cannot say', xy=(0.69, 0), xycoords='data',
                    fontsize=20, ha='center', va='bottom',
                    bbox=dict(boxstyle='square', fc='white'),
                    arrowprops=dict(arrowstyle='-[, widthB=15.1, lengthB=1', lw=1.0))


    ax.legend((allrects[0][0], allrects[1][0], allrects[2][0]),\
              ('dist = 1', 'dist = 2', 'all dists'), loc = (0.5,0.5), fontsize=25)
    ax.set_xticks(ind-width/2)
    ax.set_xticklabels(('BE, aligned', 'not BE, not aligned', 'BE, not aligned', 'not BE, aligned', 'no effect, undefined', 'BE, undefined', 'not BE, undefined',\
                        'no effect, aligned', 'no effect, not aligned'), fontsize = 25, rotation = 45)
    ax.set_ylabel('Fraction of edges', fontsize=25)
