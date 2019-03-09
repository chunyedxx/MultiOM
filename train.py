import config
import models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
con = config.Config()
con.set_in_path("E:\\newontomap_1\\benchmarks\DXX\DXX_UQU")
con.set_train_times(1000)
con.set_batches(100)
con.set_alpha(0.01)
con.set_ent_dimension(50)
con.set_margin(0.0)
con.set_bern(0)
con.set_ent_neg_rate(0)
con.set_opt_method("SGD")
con.set_export_files("./res/6.tf", 0)
con.set_out_files("./res/6.embedding.vec.json")
con.init()
con.set_model(models.Ontomap)
con.run()

