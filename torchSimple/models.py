import time
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn import metrics


class TorchClassifier:
    def __init__(self):
        self.batch_size = 1024

        self.save_folder = './Model_Weights'
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def estimate(self, y_test, y_pred):
        acc = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_test, y_pred=y_pred, average='macro')
        recall = metrics.recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        f1 = metrics.f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        return acc, precision, recall, f1

    def set_load_weights(self, path='model_weights.pt'):
        return os.path.join(self.save_folder, path)

    def set_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def set_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-8)

    def set_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def minibatch(self, x, y, mode='train'):

        dataset = torch.utils.data.TensorDataset(x, y)
        if mode == 'train':
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                               shuffle=True)
        if mode == 'eval':
            return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                               shuffle=False)

    def inplace_relu(self, m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace = True

    def toTensor(self, x):
        if type(x) is np.ndarray:
            return torch.tensor(x)
        else:
            return x

    def fit(self, x_train, y_train, x_test, y_test,
            model, epochs=5):

        # check whether use multi gpus.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        # load weights
        load_weights = self.set_load_weights()
        if os.path.exists(load_weights):
            model.load_state_dict(torch.load(load_weights))

        model.apply(self.inplace_relu)

        x_train = self.toTensor(x_train)
        x_test = self.toTensor(x_test)
        y_train = self.toTensor(y_train)
        y_test = self.toTensor(y_test)

        # create dataloader
        trainloader = self.minibatch(x_train, y_train, mode='train')
        testloader = self.minibatch(x_test, y_test, mode='eval')

        # opt
        criterion = self.set_criterion()
        optimizer = self.set_optimizer(model)
        scheduler = self.set_scheduler(optimizer)

        best_f1 = 0
        start = time.time()
        for epoch in range(epochs):
            # TRAIN
            print('-' * 20)
            print('EPOCH:{}/{}'.format(epoch + 1, epochs))
            model.train()
            training_losses, training_preds_list, training_trues_list = 0, [], []  # Upate: loss, preds, trues
            tk0 = tqdm(trainloader, total=int(len(trainloader)))  # add progress bar

            for bi, d in enumerate(tk0):
                # train model
                inputs = d[0].to(device, dtype=torch.float)  # X
                labels = d[1].to(device, dtype=torch.int64)  # y

                outputs = model(inputs)  # pred
                loss = criterion(outputs, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_losses += loss  # add loss
                training_preds_list.append(torch.argmax(outputs, 1).cpu().numpy())  # preds
                training_trues_list.append(labels.cpu().numpy())  # true

                tk0.set_postfix(loss=loss)  # update progress bar

            # update learning rate.
            scheduler.step()

            train_preds = np.concatenate(training_preds_list)
            train_trues = np.concatenate(training_trues_list)
            train_acc, train_precision, train_recall, train_f1 = self.estimate(y_test=train_trues, y_pred=train_preds)

            # TEST
            model.eval()
            testing_losses, testing_preds_list, testing_trues_list = 0, [], []  # Upate: loss, preds, trues
            tk1 = tqdm(testloader, total=int(len(testloader)))  # add progress bar

            for bi, d in enumerate(tk1):
                inputs = d[0].to(device, dtype=torch.float)  # X
                labels = d[1].to(device, dtype=torch.int64)  # y

                with torch.no_grad():
                    outputs = model(inputs)  # pred
                loss = criterion(outputs, labels)

                testing_losses += loss  # add loss
                testing_preds_list.append(torch.argmax(outputs, 1).cpu().numpy())  # preds
                testing_trues_list.append(labels.cpu().numpy())  # trues

                tk0.set_postfix(loss=loss)  # update progress bar

            test_preds = np.concatenate(testing_preds_list)
            test_trues = np.concatenate(testing_trues_list)
            test_acc, test_precision, test_recall, test_f1 = self.estimate(y_test=test_trues, y_pred=test_preds)

            print("Traning losses:{}; Accuary:{}, Precision:{}, Recall:{}, F1:{}".
                  format(training_losses / len(tk0),
                         round(train_acc, 3), round(train_precision, 3),
                         round(train_recall, 3), round(train_f1, 3)))

            print("Testing losses:{}; Accuary:{}, Precision:{}, Recall:{}, F1:{}".
                  format(testing_losses / len(tk1),
                         round(test_acc, 3), round(test_precision, 3),
                         round(test_recall, 3), round(test_f1, 3)))
            # save the best parameters
            if best_f1 < test_f1:
                best_f1 = test_f1
                save_weights = os.path.join(self.save_folder, '{}_{}.pt'.
                                            format('model_weights', str(round(best_f1, 3))))
                torch.save(model.state_dict(), save_weights)
                print('Now best f1 score is: {}, save the weights parameters'.format(round(best_f1, 3)))

            # clear gpu memory
            torch.cuda.empty_cache()

        end = time.time()
        print('Time consuming:{}'.format(end - start))

    def predict(self, x_test, model, load_weights=None):

        if load_weights is None:  # is None, load self.set_load_weights().
            load_weights = self.set_load_weights()

        # numpy to tensor, and create dataloader
        x_test = self.toTensor(x_test)
        testloader = torch.utils.data.DataLoader(x_test, batch_size=self.batch_size, shuffle=False)

        # multi GPUs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        # load weights
        model.load_state_dict(torch.load(load_weights))
        model.apply(self.inplace_relu)

        # start running.
        model.eval()
        testing_losses, testing_preds_list, testing_trues_list = 0, [], []  # Upate: loss, preds, trues
        tk0 = tqdm(testloader, total=int(len(testloader)))  # add progress bar

        for bi, d in enumerate(tk0):
            inputs = d.to(device, dtype=torch.float)  # X

            with torch.no_grad():
                outputs = model(inputs)  # pred

            testing_preds_list.append(torch.argmax(outputs, 1).cpu().numpy())  # preds

        return np.concatenate(testing_preds_list)