from evaluation import draw_bbs_on_img, evaluate_load

flag_pdf = False

det_file = '/mnt/data/wding/tmp/bugs/single_35/detections_test1.pkl'
data_path = \
    '/mnt/data/datasets/bugs_annotated_2014/new_separation/test/combined'
write_path = '/mnt/data/wding/tmp/bugs/single_35'
prob_threshold = 0.998
bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path, det_file, n_images=-1)
draw_bbs_on_img(bbs_dt_dict, bbs_gt_dict, data_path, write_path,
                prob_threshold=prob_threshold,
                overlap_threshold=0.5,
                plots_to_disk=True,
                data_set_name='test', ind_round=1,
                flag_pdf=flag_pdf,)

det_file = '/mnt/data/wding/tmp/bugs/single_21/detections_test1.pkl'
write_path = '/mnt/data/wding/tmp/bugs/single_21'
prob_threshold = 0.823
bbs_dt_dict, bbs_gt_dict = evaluate_load(data_path, det_file, n_images=-1)
draw_bbs_on_img(bbs_dt_dict, bbs_gt_dict, data_path, write_path,
                prob_threshold=prob_threshold,
                overlap_threshold=0.5,
                plots_to_disk=True,
                data_set_name='test', ind_round=1,
                flag_pdf=flag_pdf,)
