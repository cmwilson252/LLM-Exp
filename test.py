from colabdesign.mpnn import mk_mpnn_model
import pandas as pd

def RFdiffMPNN_AF2(sub):
    # Pulls desired number of generated sequences for MPNN to make using the RFdiffusion backbone
    num_seqs = sub["Number of Sequences MPNN"]

    # Initializes MPNN model
    mpnn_model = mk_mpnn_model()

    # Input the RFdiffusion pdb file to model
    mpnn_model.prep_inputs(pdb_filename="/cow02/rudenko/colowils/LLMExp/LLM-Exp/test_0.pdb")

    # Generate MPNN sequences 
    samples = mpnn_model.sample_parallel(num=num_seqs, batch=32)

    # Generates DF with sorted "scores" of the MPNN sequences in best to worst order and saves them to CSV file
    labels = ["score", "seqid", "seq"]
    data = [[samples [k][n] for k in labels] for n in range(num_seqs)]

    df = pd.DataFrame(data, columns=labels).round(3).sort_values("score").reset_index(drop=True)
    df.index.name = "rank"
    df.index += 1
    df.to_csv("upload/results.csv")
    
    # Collect the best 10 scores and put them in a list
    best_ten_seqMPNN = df.head(10)["seq"]
    seq_MPNN_list = best_ten_seqMPNN.tolist()