class TsConf(object):
    train_fids = [102]
    test_fids = [202]
    cmapss_url = "input/dataset.csv"
    cmapss_n_clusters=6
    ewm = 20
    target = 'RUL'
    timestamp = 'TIMECYCLE'

    groupids = ['FILEID', 'ENGINEID']
    norm_groupids = ['FILEID','CLUSTER']
    opset = ["Alt", "Mach", "TRA"]
    sequence_length = 30
    shift=30
    batch_size=30
    trim_left = True # trim left rows for each group to end exactly at the final of the group?
    random = False # using random shuffle in batch generator?

    path_checkpoint = './save/new_lstm/default'
    stateful = False # using stateful lstm?
    learning_rate = 2*10e-5  # 0.0001
    epochs = 1000
    ann_hidden = 16
    lstm_size = 48  # Number LSTM units
    num_layers = 2  # 2  # Number of layers
    alpha = 0 # regularization coef    
    restore_model = True #restore model when training
    save_model = False
    plot = False    
    
    columns_old = ["ENGINEID", "TIMECYCLE", "Alt", "Mach", "TRA", "Total temp at fan in (T2)", "Total temp at LPC out (T24)", "Total temp at HPC out (T30)", "Total temp at LPT out (T50)", 
    "Pres at fan in (P2)", "Total pres in bypass-duct (P15)", "Total pres at HPC out (P30)", "Physical fan speed (Nf)", 
    "Physical core speed (Nc)", "Engine pres ratio (epr=P50/P2)", "Static pres at HPC out (Ps30)", "Ratio of fuel flow to Ps30 (phi)",
    "Corrected fan speed (NRf)", "Corrected core speed (NRc)", "Bypass Ratio (BPR)", "Burner fuel-air ratio (farB)", 
    "Bleed Enthalpy (htBleed)", "Demanded fan speed (Nf_dmd)", "Demanded corrected fan speed (PCNfR_dmd)", "HPT coolant bleed (W31)",
    "LPT coolant bleed (W32)", "FILEID","RUL"]
    columns_new = ["FILEID","ENGINEID", "TIMECYCLE", "Alt", "Mach", "TRA", "Total temp at fan in (T2)", "Total temp at LPC out (T24)", "Total temp at HPC out (T30)", "Total temp at LPT out (T50)", 
    "Pres at fan in (P2)", "Total pres in bypass-duct (P15)", "Total pres at HPC out (P30)", "Physical fan speed (Nf)", 
    "Physical core speed (Nc)", "Engine pres ratio (epr=P50/P2)", "Static pres at HPC out (Ps30)", "Ratio of fuel flow to Ps30 (phi)",
    "Corrected fan speed (NRf)", "Corrected core speed (NRc)", "Bypass Ratio (BPR)", "Burner fuel-air ratio (farB)", 
    "Bleed Enthalpy (htBleed)", "Demanded fan speed (Nf_dmd)", "Demanded corrected fan speed (PCNfR_dmd)", "HPT coolant bleed (W31)",
    "LPT coolant bleed (W32)", "RUL"]
    
    select_feat = ["Total temp at LPC out (T24)", 
               "Total temp at HPC out (T30)", 
               "Total temp at LPT out (T50)", 
               "Physical core speed (Nc)", 
               "Static pres at HPC out (Ps30)", 
               "Corrected core speed (NRc)", 
               "Bypass Ratio (BPR)", 
               "Bleed Enthalpy (htBleed)"]
    
    def getChannels(self):
        return len(self.select_feat)

    n_channels = property(getChannels)
    
    def getPathCh(self):
        return './save/new_lstm/'+'cha_'+str(self.n_channels)+'_len_'+str(self.sequence_length) \
                +'_bat_'+str(self.batch_size) +'/ds'+str(self.train_fids[0])+'_lstm_'+str(self.num_layers) \
                +'_layers_stateful_'+str(self.stateful)

    path_checkpoint = property(getPathCh)    
    
    def __init__(self):
        """Set values of computed attributes."""
        self.features = [x for x in self.columns_new if x not in (self.groupids+[self.timestamp]+[self.target]+self.opset)]        
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

