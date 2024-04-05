import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def add_prediction_subplot(date, y_pred, y_true, ax, data_freq='10min', plot_date=None, xlabel=None, ylabel=None, title=None):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df.index = pd.to_datetime(date)
    df = df.resample(data_freq).asfreq()
    df = df.reset_index(names='date')

    if plot_date is not None:
        df_sub = df[df['date'].dt.date == plot_date]
    else:
        df_sub = df
    ax.plot(df_sub['date'], df_sub['y_true'], label='true y')
    ax.plot(df_sub['date'], df_sub['y_pred'], label=f'model prediction')
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(f"{plot_date}")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    formatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    

def plot_losses(model_names, loss_dicts, best_epochs, fig_path=None, plot_format='pdf'):
    for mname, loss_dict, i in zip(model_names, loss_dicts, best_epochs):
        x = np.arange(len(loss_dict['train_loss']))
        fig, ax = plt.subplots(2,3,figsize=(14,5), sharey='row', sharex=True)
        ax[0,0].plot(x, loss_dict['train_loss'], color='lightseagreen', label='train')
        ax[0,0].axvline(i, linestyle='--', color='grey', lw=.5)
        ax[0,0].set_title(f'{mname}, Loss')
        ax[0,0].legend()

        ax[0,1].plot(x, loss_dict['train_loss_pred'], color='lightseagreen', label='train')
        ax[0,1].axvline(i, linestyle='--', color='grey', lw=.5)
        ax[0,1].set_title(f'{mname}, Pred loss')
        ax[0,1].legend()

        if loss_dict['train_loss_phys'][0] != 0:
            ax[0,2].plot(x, loss_dict['train_loss_phys'], color='lightseagreen', label='train')
            ax[0,2].axvline(i, linestyle='--', color='grey', lw=.5)
            ax[0,2].set_title(f'{mname}, Phys loss')
            ax[0,2].legend()

        ax[1,0].plot(x, loss_dict['val_loss'], color='coral', label='val')
        ax[1,0].axvline(i, linestyle='--', color='grey', lw=.5)
        ax[1,0].legend()

        ax[1,1].plot(x, loss_dict['val_loss_pred'], color='coral', label='val')
        ax[1,1].axvline(i, linestyle='--', color='grey', lw=.5)
        ax[1,1].legend()


        if loss_dict['val_loss_phys'][0] != 0:
            ax[1,2].plot(x, loss_dict['val_loss_phys'], color='coral', label='val')
            ax[1,2].axvline(i, linestyle='--', color='grey', lw=.5)
            ax[1,2].set_ylim(0, max(loss_dict['val_loss_phys']))
            ax[1,2].legend()

        fig.tight_layout()
        if fig_path is not None:
            plt.savefig(Path(fig_path, f"loss_{mname}"), format=plot_format, dpi=300)
            plt.close()
        else:
            plt.show()
