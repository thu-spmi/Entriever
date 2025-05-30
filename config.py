
import logging, time, os

class _Config:
    def __init__(self):
        self.seed=6
        self.exp_name='temp'
        self.exp_path=''
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.mode='train'

        #self.gpt_path ='uer/gpt2-chinese-cluecorpussmall'#can not connected#can not connected
        #self.posterior_path = 'uer/gpt2-chinese-cluecorpussmall'#can not connected
        # 'experiments/pretrain1/best_model'
        self.gpt = True
        self.gpt_path = 'gpt2-medium'
        self.posterior_path ='gpt2-medium'
        # experiments/pretrain/best_model pretrained model with extra special token, can be modified to fit the finetuning procedure
        # experiments/pretrain1/best_model pretrained model without extra special token
        self.t5_path = 'experiments/pretrain_t5/best_model'
        self.t5_posterior_path ='experiments/pretrain_t5/best_model' # 'uer/t5-base-chinese-cluecorpussmall'
        # 'uer/t5-v1_1-base-chinese-cluecorpussmall'
        self.data_path='data/processed_data.json'
        self.data_dir='data/'

        self.device=[0]
        self.batch_size=8
        self.origin_batch_size=8
        self.gradient_accumulation_steps=4
        self.epoch_num=40
        self.eval_batch_size=32
        self.lr = 2e-5
        self.warmup_ratio=0.2
        self.pad_id=0
        self.only_target_loss=True
        self.save_type='max_score'
        #self.save_type='min_loss'
        self.debugging=False

        self.no_user_intent = True # change to simple structure
        self.kb_grounding = False # whether to put the kb in the front or not
        self.db_change = False # change db to kb (all of the local kb)
        self.act_change = False # change act to appear in front of kb to enable proactive questioning
        self.dst = False # remain to be added
        self.with_context = False # add context before the input
        self.gt_db = True # using ground truth db i.e slot value pairs that appear in the next utterance
        self.ground_truth = False # using ground truth user intent to query db
        self.posterior_change = False # need to be removed in the future
        self.retrieve_kb = False # test config for retrieval based model

        # config for retrieval augment settings
        self.joint_training = False
        self.fix_retrieval_model = True
        self.mix_retrieval_training = False

        # config for retrieval model
        
        self.train_retrieve = False
        self.fine_grained = False
        self.bert_path ='bert-base-chinese'
        self.bert_path_en ='bert-base'
        self.context_save_path ='ret_exp/best_context'
        self.triple_save_path ='ret_exp/best_triple'
        self.retrieval_save_path ='ret_exp/best_retreive' # two bert models with a bilinear layer
        self.bert_save_path ='ret_exp/best_bert_fg1' if self.fine_grained else 'ret_exp/best_bert_old'

        self.max_sequence_len =512
        self.only_one_model = True
        self.retrieve_hist = 3
        self.bert_save_path = self.bert_save_path + str(self.retrieve_hist)
        # self.bert_save_path_en = self.bert_save_path_en + str(self.retrieve_hist)
        
        # config for pseudo label:
        self.with_pseudo_label = False
        self.pseudo_path = 'data/encoded_pseudo_data.json'
        self.pseudo_porportion = 2
        self.pseudo_label_retrieval = False
        if self.pseudo_label_retrieval:
            self.bert_save_path = self.bert_save_path + '_pseudo'

        # config for ebm
        self.dropout = 0.03
        self.ebm_save_path = 'ret_exp/best_ebm'
        self.train_sample_times = 1
        self.train_sample_num = 12*self.train_sample_times # sample num for importance sampling, 12 is negative sample batch-size
        # self.test_sample_num = 200 # sample num for importance sampling, discard, do not use sample in testing
        self.test_num = 16 # top-k for rescoring 
        self.train_ebm = False
        self.train_ebm_mis = False
        self.residual = False
        self.reject_control = False
        self.use_all_proposal = True
        self.add_extra_feature = False
        self.reg_weight = 1.0
        self.rescore_for_generation = False
        
        # config for jsa
        # unsupervise data proportion in jsa
        self.jsa_unsup = 1
        self.jsa_ebm = False
        self.jsa_ebm_joint = False

        # config for generation
        self.temperature = 1.0
        self.threshold = 0.1

    def _init_logging_handler(self, dataset=None):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if 'train' in self.mode:
            if dataset:
                filename = './log/log_{}_{}_sd{}_{}.txt'.format(self.mode, self.exp_name, self.seed, dataset)
            else:
                filename = './log/log_{}_{}_sd{}.txt'.format(self.mode, self.exp_name, self.seed)
            file_handler = logging.FileHandler(filename)
        else:
            file_handler = logging.FileHandler(os.path.join(self.gpt_path, 'eval_log.txt'))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()