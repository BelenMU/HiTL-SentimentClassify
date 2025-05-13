import numpy as np
import scipy.io as sio
from scipy.linalg import svd
import time
import select_first_word_Hsubstraction as selectFirst
import select_next_word_Hsubstraction as selectNext
import get_expected_score as get
import update_given_y_logistic as updateY
import update_given_x as updateX

start_time = time.time()

# Parameters to set
filename = "results_v2.mat"  # Directory in which to save results
num_iterations = 10  # Number of interactions with humans
num_initializations = 1  # Number of experiments
num_S = 2  # Number of words in query, |S|
word_selection = True  # Whether to include word selection in query
active = True  # Whether to include active word set selection

# Load dataset from SocialSent and its parameters
mat_data = sio.loadmat(r'C:\Users\yslee\OneDrive\Desktop\Research\main\init_SocialSent_freq.mat')

mu_init = mat_data['mu_init']
mu_init = mu_init.flatten()
C_init = mat_data['C_init']
theta = mat_data['theta']
noise_factor = mat_data['noise_factor']
scale = mat_data['scale']
list_embeddings = mat_data['list_embeddings']
list_score = mat_data['list_score']
list_var = mat_data['list_var']
xtrain = mat_data['xtrain']
y_train = mat_data['y_train']
d = mat_data['d'].item()  # Assuming d is a scalar

mu = np.array([-0.000205393239041505,
-0.0131726904357769,
-0.0273435313032181,
-0.00865487005364662,
-0.0937814602174014,
-0.0441635284450941,
0.00635864632271388,
-0.0743620812449436,
0.0196236518242190,
0.0607673632540245,
0.0144678201556068,
-0.110126692078021,
-0.00360980410544739,
0.00557484804011332,
0.0180962005842886,
0.0853697126746907,
-0.0428396811690980,
-0.0375755198491427,
0.0199810782058240,
0.0361833099172749,
-0.0971441431069387,
0.0249380056123051,
-0.0552220412801171,
0.0801024974357563,
0.0129398046743798,
0.0102447235192270,
0.115480968261082,
-0.0559557325606131,
-0.0504669948091636,
-0.0213298743935478,
0.0364981128222055,
0.0257652992552550,
-0.0317325405370301,
0.0941301271158118,
-0.0461053748820584,
-0.132474969426033,
-0.0316447624067545,
0.00455181770548144,
-0.0162471293867563,
0.101196768614913,
0.00767769454611247,
0.0603356453624456,
-0.104171725494056,
-7.85572607938496e-05,
-0.0351015451149503,
0.0267813674247997,
-0.0258261785713811,
-0.0183428935564167,
0.0152793509946721,
-0.0284367526777905,
0.128098119261472,
-0.158750311355074,
-0.0287342752239669,
-0.0477269968393036,
-0.00937534362052504,
0.0660020569766844,
-0.0695446411618435,
-0.0631343297345406,
0.0844370080967488,
0.0218357289119960,
0.0647537959125849,
0.0494311588375177,
0.0707398393794744,
-0.0103377135396187,
0.0116674365665662,
0.0595714221954388,
0.0503047845121647,
-0.0155503644258878,
-0.0337999317093688,
-0.0756162247943099,
0.0551280282270545,
0.00279756942452896,
-0.0600943048672785,
0.00177317438844633,
-0.0385064577054511,
0.00204348080318033,
-0.0410841304162830,
-0.0488188045156895,
-0.00477652111377805,
0.0250792368388692,
-0.0653900447489513,
0.0788174400780169,
0.0512473616963516,
0.0791014888238300,
0.0609722447048556,
-0.0395959134026808,
-0.0218798982245130,
0.000567782071681908,
-0.0805988011405746,
-0.0518436314729266,
0.0827688580704322,
0.0469370626274249,
0.0545168932284203,
0.0278753403125643,
-0.0548923291638765,
0.0256375733861250,
-0.0906082753010121,
0.0610437844499417,
-0.0709410904859528,
0.000485448245940064,
0.0172596292035191,
0.122100291730118,
0.0632942212968062,
0.0439398456588649,
0.0182280955206249,
-0.0133246651464025,
-0.0141078414477067,
0.0487666005864705,
-0.0782001175673359,
-0.0775845127838863,
-0.00601650292342312,
0.00441355258712557,
-0.0375497256305215,
0.0303959807031256,
-0.00232620103722267,
0.0110563016861895,
-0.0395292280414662,
0.0528535939923276,
-0.0900765475462476,
0.0374118989446822,
0.0460203198992070,
-0.0148312028548034,
0.0251383862907760,
-0.0600028781629953,
-0.00703804593956036,
-0.0522285570094513,
0.0496968946783509,
0.0310441027737595,
-0.0466594809227631,
-0.170060267105266,
-0.0237357414870743,
-0.0648672691486836,
-0.00372931686262660,
-0.0409607141114344,
-0.0321197479387916,
-0.0335575585309482,
0.0672241395154389,
0.0636003421353213,
-1.31465209965393e-05,
0.0410284445210216,
-0.0427140492870763,
-0.0388664814068530,
0.0451636829525716,
0.128211565822140,
-0.0865573774687002,
-0.0440600487801794,
0.00892142033377567,
-0.0709621048719831,
0.0743875755623324,
-0.0407152725426310,
0.0414485467131867,
-0.165298159816929,
0.0431966563711370,
0.000655701506758192,
0.0329260156882321,
-0.0131881289783517,
0.0169348600391268,
0.0197126109337342,
0.134919426359994,
0.0535526203257423,
-0.0437990511957948,
-0.00455818848612936,
0.0479680214441568,
-0.0285732563640798,
-0.0929313920839509,
-0.0629676049257810,
-0.0432116579579670,
-0.00359127369452084,
0.0327977559030277,
-0.0316765314121882,
-0.0119756999541920,
0.0207966371773249,
-0.0577628557780872,
-0.0197892084042766,
-0.0242557927515993,
-0.0310860087035079,
0.0498286985276786,
-0.0790962693645753,
-0.0630019349501224,
-0.0681209619013127,
0.0211073626552819,
-0.121929529886420,
0.0318913812228562,
-0.129127003640087,
0.0225595023977306,
-0.0224047875134068,
0.0145999432847306,
0.0870661825504474,
-0.0179588344650640,
-0.0206679248177697,
0.0533495071542503,
0.0188020174604004,
-0.0887920809098221,
0.0129304220360770,
0.0164305415578386,
0.0300477829921333,
0.0499520416061463,
0.0242809607962151,
-0.0907384994995169,
-0.0641553801645083,
0.0341048307780034,
-0.0778148733004000,
0.130809358789880,
0.0903048949090624,
-0.0654738576701195,
0.0356985109014550,
0.0338695100941851,
0.0863272356985571,
0.00249530977778580,
-0.0137039117634560,
0.0995709675442590,
-0.0301037537633062,
-0.0711453420851309,
0.0363236350844043,
0.0617689480343991,
0.0249309016124480,
0.00512746220969069,
-0.0433782233077098,
-0.00132407227786276,
-0.0105055156714143,
-0.0229413804270999,
0.0739569592572422,
-0.0487944439051730,
-0.0708874817069972,
-0.0205229848526899,
0.126661736558882,
-0.0550362844910087,
-0.0107148626725642,
-0.0185092343159107,
0.0307840395479768,
0.0167129396700926,
0.0198084092948550,
-0.0680024961497852,
0.0304267443681441,
-0.161334631683367,
0.0661512231681875,
0.0303227567034138,
0.0547316198303374,
-0.0525177808682420,
-0.143841422857815,
-0.0271793743851450,
-0.0320843345666949,
-0.0285561784827923,
-0.00977221367570313,
-0.00245176914095561,
0.0720075998080528,
0.00700398373180951,
0.0467093638218989,
0.0624637275061360,
0.111555358101763,
0.0275281444589235,
-0.0548459499067203,
-0.0651914232212998,
0.0200027241386033,
0.0451111282464232,
-0.0298711189490057,
-0.0231326888658629,
0.0204240047142727,
0.107843626215598,
-0.0472250806956142,
0.0471906649434714,
-0.149740382931920,
0.0251969191918971,
-0.0548925989161351,
-0.125063674352180,
-0.0114279395542580,
-0.0640645782000740,
-0.000195204097999918,
0.00260838849484081,
0.0126304621487207,
-0.00168036317079818,
0.0122900755714265,
-0.0336707558419991,
0.0294238620482352,
-0.0117976038448614,
-0.105922838221470,
0.0529214468589673,
0.0116076970024865,
-0.0241638516909456,
-0.0232795456948517,
-0.0218379873980084,
0.0409459900672291,
0.0682221074124860,
-0.000103458031471618,
-0.0245705791937606,
0.0310192190683751,
0.0871533694988787,
0.0317080789623868,
0.0442185074545636,
0.00373231778733875,
0.0334231130899997,
0.000621898422032377,
0.0641982402815592,
0.0843701106500657,
0.0728199257228074,
0.117068241393377,
-0.0341881606400797,
-0.0784850684082916,
0.0296660561141888,
-0.00648498293574970,
-0.0861811933715622])

end_mu = np.zeros((d+1, num_initializations))
end_sigma = np.zeros((d+1, d+1, num_initializations))
mu_save = np.zeros((num_initializations, d+1, num_iterations))

C_root_det = np.zeros((num_iterations+1, num_initializations))
MSE_MT_init = np.zeros((num_iterations+1, num_initializations))
accu_init = np.zeros((num_iterations+1, num_initializations))
max_score_init = np.zeros((num_iterations, num_initializations))

if not word_selection:
    num_S = 1
    active = False

# Run algorithm
for init in range(num_initializations):
    print("INITIALIZATION")
    # mu = np.random.multivariate_normal(mu_init, C_init).T # mu: Python: (301,), Matlab: (301 x 1)
    mu_temp = mu
    mu = mu.reshape(-1, 1)
    sigma = C_init # sigma: Python: (301, 301), Matlab: (301 x 301)
    C_root_det_temp = np.zeros(num_iterations+1) # C-root_det_temp: Matlab: (11, 1) Python: (11,)
    MSE_MT_init_temp = np.zeros(num_iterations+1) # MSE_MT_init_temp: Matlab: (11, 1) Python: (11,)
    accu_init_temp = np.zeros(num_iterations+1) # accu_init_temp: Matlab: (11, 1) Python: (11,)
    C_root_det_temp[0] = np.prod(np.power(np.linalg.svd(sigma, compute_uv=False), 1/(d+1))) # OK, Value matches
    MSE_MT_init_temp[0] = np.linalg.norm(theta - mu)**2 # OK, Value matches
    accu_init_temp[0] = 1 - np.sum(np.abs(np.sign(mu.T @ xtrain) - y_train)) / (2 * len(y_train[0])) # OK, Value matches
    max_score = 0 # OK
    mu = mu_temp
    for ii in range(num_iterations):
        mu_save[init, :, ii] = mu # mu : (301,)
        # query_selected = np.random.randint(1, 3)
        query_selected = 1
        if active:
            S, ind_S, list_embeddings_remaining = selectFirst.select_first_word_Hsubstraction(list_embeddings, mu, sigma, noise_factor)
            for loop in range(1, num_S):
                S, ind_S, list_embeddings_remaining, max_score = selectNext.select_next_word_Hsubstraction(
                    S, ind_S, list_embeddings_remaining, mu, sigma, noise_factor, scale, query=query_selected
                )
            max_score_init[ii, init] = max_score
        else:
            ind_S = np.random.permutation(len(list_score))[:num_S]
            S = list_embeddings[:, ind_S]
            EH, HE = get.get_expected_score(S, mu, sigma, noise_factor, scale, query_selected=query_selected)
            max_score_init[ii, init] = HE - EH
        # Not sure about the index
        selected = np.array([[-0.2476], [-1.2402]])
        sample_score = list_score[ind_S] + selected * np.sqrt(list_var[ind_S])
        if query_selected == 1:
            score_x = np.max(sample_score)
            ind_x = np.argmax(sample_score)
        else:
            score_x = np.min(sample_score)
            ind_x = np.argmin(sample_score)
        y = 1 if score_x > 0 else -1
        x_t = S[:, ind_x]
        mu, sigma = updateY.update_given_y_logistic(x_t * noise_factor, y, sigma, mu)

        if word_selection:
            if query_selected == 1:
                mu, sigma = updateX.update_given_x(mu, sigma, scale * S, ind_x)
            else:
                mu, sigma = updateX.update_given_x(mu, sigma, -1 * scale * S, ind_x)
        C_root_det_temp[ii+1] = np.prod(np.power(np.linalg.svd(sigma, compute_uv=False), 1/(d+1)))
        temp = mu.T.reshape(-1, 1)
        MSE_MT_init_temp[ii+1] = np.linalg.norm(theta - temp)**2
        accu_init_temp[ii+1] = 1 - np.sum(np.abs(np.sign(mu.T @ xtrain) - y_train)) / (2 * y_train.shape[1])

        print(f'Iteration {ii}/{num_iterations} in initialization {init}')
    
    C_root_det[:, init] = C_root_det_temp
    MSE_MT_init[:, init] = MSE_MT_init_temp
    accu_init[:, init] = accu_init_temp
    end_mu[:, init] = mu
    end_sigma[:, :, init] = sigma

# Save the results
sio.savemat(filename, {
    'end_mu': end_mu,
    'end_sigma': end_sigma,
    'mu_save': mu_save,
    'C_root_det': C_root_det,
    'MSE_MT_init': MSE_MT_init,
    'accu_init': accu_init,
    'max_score_init': max_score_init
})

end_time = time.time()

print("Done. Time elapsed: ", end_time - start_time, " seconds")