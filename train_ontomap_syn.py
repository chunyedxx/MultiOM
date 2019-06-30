import config
import models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
con = config.Config()
con.set_in_path(".\\Datasets\DXX\DXX_SYN")
con.set_train_times(1000)
con.set_batches(10)
con.set_alpha(0.01)
con.set_ent_dimension(50)
con.set_negative_sampling('unif')
con.set_ent_neg_rate(5)
con.set_opt_method("SGD")
con.set_export_files("./res/syn0622.tf", 0)
con.set_out_files("./res/syn0622.embedding.vec.json")
con.model_name("ontomap_syn")
con.init()
con.set_model(models.Ontomap_syn)
con.run()
