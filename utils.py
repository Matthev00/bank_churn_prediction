import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Bank Churn Modelling using XGBoost')

    parser.add_argument('--data_path', type=str, default='Churn_Modelling.csv')
    parser.add_argument('--run_name', type=str, default='Best Model')
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--reg_lambda', type=float, default=0)
    parser.add_argument('--scale_pos_weight', type=int, default=1)

    return parser.parse_args()
