import torch
from torch import nn
from tqdm import tqdm

class RNNCell(nn.Module):
    """Standard LSTM Cell"""

    def __init__(self, input_dims, hidden_dims, output_dim=1, num_rnn_layers=1):
        super().__init__()
        self.num_layers = num_rnn_layers
        self.hidden_dims = hidden_dims
        self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers=num_rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        if x.ndim == 2:
            x_in = x.view(x.shape[0], -1, 4)
        else:
            x_in = x
        h0 = torch.rand(self.num_layers, x_in.shape[0], self.hidden_dims).to(x_in.device)
        c0 = torch.rand(self.num_layers, x_in.shape[0], self.hidden_dims).to(x_in.device)
        out, (h, c) = self.lstm(x_in, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
    
    def predict(self, test_data):
        self.eval()
        with torch.no_grad():
            predictions = self(test_data)
        return predictions
    

class PCRNN(nn.Module):
    """Physics-constrained wrapper for RNNCell"""

    def __init__(self, layers, R=1.0, lambda1=1.0, lambda2=1.0, alpha=1):       
        super(PCRNN, self).__init__()

        self.layers = layers

        # Physics parameters (log to ensure learning of positive parameter values)
        self.R = nn.Parameter(torch.log(torch.tensor(R, dtype=float)), requires_grad=True)
        self.lambda1 = nn.Parameter(torch.log(torch.tensor(lambda1, dtype=float)), requires_grad=True)
        self.lambda2 = nn.Parameter(torch.log(torch.tensor(lambda2, dtype=float)), requires_grad=True)
        self.alpha = alpha

    def forward(self, input):
        output = self.layers(input)
        return output

    def net_u(self, inputs):
        Tb_pred = self(inputs)
        return Tb_pred

    def net_f(self, temp_bearing_t1_predicted, state_t1):
        power_t1 = state_t1[:, -1, 0:1]
        temp_ambient_t1 = state_t1[:, -1, 1:2]
        rotor_speed_t1 = state_t1[:, -1, 2:3]
        temp_bearing_t0 = state_t1[:, -1, 3:]

        dT_pred = self._temp_grad_euler(temp_bearing_t1_predicted, temp_bearing_t0)
        dT_comp = self._temp_grad_computed(temp_bearing_t0, temp_ambient_t1, rotor_speed_t1, power_t1)
            
        return dT_pred, dT_comp

    def _temp_grad_euler(self, temp_bearing_t1, temp_bearing_t0):
        grad = temp_bearing_t1 - temp_bearing_t0
        return grad
    
    def _temp_grad_computed(self, temp_bearing_t0, temp_ambient_t1, rotor_speed_t1, power_t1):
        R = torch.exp(self.R)
        lambda1 = torch.exp(self.lambda1)
        lambda2 = torch.exp(self.lambda2)

        grad = R * (temp_ambient_t1 - temp_bearing_t0) + lambda1 * (rotor_speed_t1) + lambda2 * (power_t1)
        return grad

    def predict(self, test_data):
        self.eval()
        with torch.no_grad():
            predictions = self(test_data)
        return predictions
        
    def get_gradients(self, test_data):
        power_t1 = test_data[:, -1, 0:1]
        temp_ambient_t1 = test_data[:, -1, 1:2]
        rotor_speed_t1 = test_data[:, -1, 2:3]
        temp_bearing_t0 = test_data[:, -1, 3:]
            
        with torch.no_grad():
            grad_computed = self._temp_grad_computed(temp_bearing_t0, temp_ambient_t1, rotor_speed_t1, power_t1)
        return grad_computed


def train(net, optimizer, criterion, num_epochs, train_set, validation_set=None, early_stopping=None, device=None, disable_output=False):
    if early_stopping and not validation_set:
        raise AssertionError("validation_set is needed for early_stopping.")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_dict = {
        'train_loss': [],
        'train_loss_pred': [],
        'train_loss_phys': [],
        'val_loss': [],
        'val_loss_pred': [],
        'val_loss_phys': []
        }
    
    pbar = tqdm(range(num_epochs), bar_format=f"{{percentage:3.0f}}%|{{bar}}{{r_bar}}{{desc}}", disable=disable_output)
    for epoch in pbar:
        running_loss = 0
        running_loss_prediction = 0
        running_loss_physics = 0

        # train step
        net.train()
        num_batches_train = len(train_set)

        for batch, (X, y) in enumerate(train_set):
            outputs = net.net_u(X)
            loss_prediction = criterion(outputs, y)
            running_loss_prediction += loss_prediction.item()
            dT_pred, dT_comp = net.net_f(outputs, X)
            loss_physics = criterion(dT_pred, dT_comp)
            loss = loss_prediction + net.alpha * loss_physics
            running_loss_physics += loss_physics.item()
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                desc = f" loss: {running_loss / 100}  (pred: {running_loss_prediction / 100}, phys: {running_loss_physics / 100})"
                if validation_set:
                    if epoch == 0:
                        val_loss = 0
                    desc += f"  val_loss: {val_loss}"
                pbar.set_description_str(desc)
                running_loss = 0
                running_loss_prediction = 0
                running_loss_physics = 0

        loss_dict['train_loss'].append(running_loss / num_batches_train)
        loss_dict['train_loss_pred'].append(running_loss_prediction / num_batches_train)
        loss_dict['train_loss_phys'].append(running_loss_physics / num_batches_train)

        # validation step
        if validation_set:
            num_batches_test = len(validation_set.dataset)
            val_loss = 0
            val_loss_prediction = 0
            val_loss_physics = 0

            net.eval()
            with torch.no_grad():
                for X, y in validation_set:
                    pred = net.net_u(X)
                    l_pred = criterion(pred, y).item()
                    val_loss_prediction += l_pred
                    dT_pred, dT_comp = net.net_f(pred, X)
                    l_phys = criterion(dT_pred, dT_comp).item()
                    val_loss += (l_pred + net.alpha * l_phys)
                    val_loss_physics += l_phys

            val_loss /= num_batches_test
            val_loss_prediction /= num_batches_test
            val_loss_physics /= num_batches_test
            loss_dict['val_loss'].append(val_loss)
            loss_dict['val_loss_pred'].append(val_loss_prediction)
            loss_dict['val_loss_phys'].append(val_loss_physics)

        if early_stopping:
            best_net = early_stopping.evaluate(val_loss, net)
            if best_net:
                print(f"Early Stopping: Stopped training after epoch {epoch + 1}, restoring model from epoch {early_stopping.best_model_epoch}")
                return best_net, loss_dict, early_stopping.best_model_epoch
           
    if early_stopping and early_stopping.best_model_epoch < (epoch + 1):
        best_net = early_stopping.restore_model()
        if best_net is not None:
            print(f"Early Stopping: Training finished, restoring model from epoch {early_stopping.best_model_epoch}")
            return best_net, loss_dict, early_stopping.best_model_epoch
        else:
            return net, loss_dict, epoch + 1
    else:
        return net, loss_dict, epoch + 1