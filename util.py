import os
import pandas as pd
import matplotlib.pyplot as plt


def unzip(zipped):
    data1, data2 = zip(*zipped)
    return list(data1), list(data2)


def plot_and_save_performance(exact_performance, apprx_performance):
    print('Saving performance...', end='', flush=True)
    exact_trn_performance, exact_tst_performance = exact_performance
    apprx_trn_performance, apprx_tst_performance = apprx_performance

    exact_trn_accr, exact_trn_loss = unzip(exact_trn_performance)
    exact_tst_accr, exact_tst_loss = unzip(exact_tst_performance)

    apprx_trn_accr, apprx_trn_loss = unzip(apprx_trn_performance)
    apprx_tst_accr, apprx_tst_loss = unzip(apprx_tst_performance)

    performance = {
        'exact_trn_accr': exact_trn_accr,
        'exact_trn_loss': exact_trn_loss,
        'exact_tst_accr': exact_tst_accr,
        'exact_tst_loss': exact_tst_loss,
        'apprx_trn_accr': apprx_trn_accr,
        'apprx_trn_loss': apprx_trn_loss,
        'apprx_tst_accr': apprx_tst_accr,
        'apprx_tst_loss': apprx_tst_loss}

    # save train and test data to csv file
    df = pd.DataFrame(performance)

    # saving the dataframe
    df.to_csv('performance.csv')
    print('done!')

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # plot network performance and save plot
    fig1 = plt.figure(figsize=(8, 6))

    plt.plot(exact_trn_accr, '.', label='EventProp Training')
    plt.plot(exact_tst_accr, '-', label='EventProp Test')
    plt.plot(apprx_trn_accr, '.', label='SNUP Training')
    plt.plot(apprx_tst_accr, '-', label='SNUP Test')
    plt.legend(loc='best')

    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy %')
    plt.title('Comparison of Model Accuracy')

    plt.tight_layout()
    plt.savefig('accuracy_plot.png')  # save network performance plot
    plt.show()

    # plot training loss and save plot
    fig2 = plt.figure(figsize=(8, 6))

    plt.plot(exact_trn_loss, '.', label='EventProp Training')
    plt.plot(exact_tst_loss, '-', label='EventProp Test')
    plt.plot(apprx_trn_loss, '.', label='SNUP Training')
    plt.plot(apprx_tst_loss, '-', label='SNUP Test')
    plt.legend(loc='best')

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Comparison of Model Loss')

    plt.tight_layout()
    plt.savefig('loss_plot.png')  # save training loss plot
    plt.show()
