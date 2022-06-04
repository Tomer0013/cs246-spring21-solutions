import numpy as np


def id_to_title_dict(text_path):
    id_to_title_dict = {}
    for idx, row in enumerate(open(text_path, encoding="utf-8")):
        id_to_title_dict[idx] = row.split("\n")[0][1:-1]
    return id_to_title_dict

def top_k_recs_for_user(scores_mat, user, idx_to_title_dict, top_k=5, keep_scores=False):
    user_preds = sorted([(idx, score) for idx, score in enumerate(scores_mat[user])], \
                 key= lambda x: (-x[1], x[0]))
    if keep_scores:
        return [(idx_to_title_dict[x[0]], x[1]) for x in user_preds][:top_k]
    return [idx_to_title_dict[x[0]] for x in user_preds][:top_k]

def print_recs_list(list_title, recs_list, empty_line_at_the_end=True):
    print(list_title)
    for idx, title in enumerate(recs_list):
        print(f"{idx+1}. {title}")
    if empty_line_at_the_end:
        print()


if __name__ == "__main__":

    # Init
    shows_path = "data/shows.txt"
    user_shows_path = "data/user-shows.txt"
    r_mat = np.loadtxt(user_shows_path)

    # Create id to title dict
    id_to_title_dict = id_to_title_dict(shows_path)

    # Compute matrices
    rt_r = r_mat.T.dot(r_mat)
    rr_t = r_mat.dot(r_mat.T)

    p = np.diag(rr_t)
    q = np.diag(rt_r)

    su_mat = ((1 / np.sqrt(p)).reshape(1, -1)) * rr_t * ((1 / np.sqrt(p)).reshape(-1, 1))
    si_mat = ((1 / np.sqrt(q)).reshape(1, -1)) * rt_r * ((1 / np.sqrt(q)).reshape(-1, 1))

    user_based_preds = su_mat.dot(r_mat)
    item_based_preds = r_mat.dot(si_mat)

    # Create user and item based CF recs for Alex
    top_5_ucf_recs_for_alex = top_k_recs_for_user(user_based_preds, 499, id_to_title_dict)
    top_5_icf_recs_for_alex = top_k_recs_for_user(item_based_preds, 499, id_to_title_dict)

    # Print recs
    print_recs_list("User based CF recs for Alex:", top_5_ucf_recs_for_alex)
    print_recs_list("Item based CF recs for Alex:", top_5_icf_recs_for_alex)
