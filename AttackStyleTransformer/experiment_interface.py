import torch
from torch.autograd import Variable
    
class ExperimentInterFace():

    def __init__(self):
        pass 

    def pretrain_G(self, G, F, G_loss, G_optim, inputs, labels, num_of_epochs, alpha, beta):
        for _ in range(num_of_epochs):
            train_loss = None
            for i in range(len(inputs)):
                data , label = inputs[i], labels[i]
                data = Variable(data)
                new_data, mu, logvar = G(data)

                new_data = new_data.reshape(-1, 1, 28, 28)
                score = F.forward(new_data)
                prediction = torch.argmax(score)
                loss, _, _, _ = G_loss(new_data, data, mu, logvar, score, label, prediction, True, 0, alpha, beta)
                
                if train_loss is None:
                    train_loss = loss
                else:
                    train_loss += loss
                
            G_optim.zero_grad()
            train_loss.backward(retain_graph=True)
            G_optim.step()

    def run(self, F, G, D, G_loss_func, D_loss_func, G_optim, D_optim, num_of_train_epochs, inputs, labels, G_alpha, G_beta):
        import torch
        import matplotlib.pyplot as plt
        from torch.autograd import Variable
        import numpy as np

        draw_points_l1, draw_points_l2, draw_points_l3, draw_points_l4 = [], [], [], []

        # Experiment: increasing G_beta (slow starting)
        one_unit_of_G_beta = G_beta

        for epoch in range(1, num_of_train_epochs+1):    
            train_loss, train_loss_l1, train_loss_l2, train_loss_l3 = None, None, None, None
            print(epoch, '/', num_of_train_epochs, ',', 'G_beta:', G_beta)
            for i in range(len(inputs)):
                data , label = inputs[i], labels[i]
                data = Variable(data)
                new_data, mu, logvar = G(data)

                # Get scores from Discriminator for both x and x'
                real_score_by_D = D(data)
                fake_score_by_D = D(new_data)

                # Adjust Discriminator 
                D_loss = D_loss_func(True, real_score_by_D)
                D_loss += D_loss_func(False, fake_score_by_D)
                D_optim.zero_grad()
                D_loss.backward(retain_graph=True)
                D_optim.step()

                new_data = new_data.reshape(-1, 1, 28, 28)
                score = F.forward(new_data)
                prediction = torch.argmax(score)

                loss, l1, l2, l3 = G_loss_func(new_data, data, mu, logvar, score, label, prediction, False, fake_score_by_D, G_alpha, G_beta)

                if train_loss is None:
                    train_loss = loss
                    train_loss_l1 = l1
                    train_loss_l2 = l2
                    train_loss_l3 = l3
                else:
                    train_loss += loss
                    train_loss_l1 += l1
                    train_loss_l2 += l2
                    train_loss_l3 += l3

            G_optim.zero_grad()
            train_loss.backward(retain_graph=True)
            G_optim.step()

            draw_points_l1.append(train_loss_l1)
            draw_points_l2.append(train_loss_l2)
            draw_points_l3.append(train_loss_l3)
            draw_points_l4.append(train_loss)

            G_beta += one_unit_of_G_beta # experimental purpose

        x = [x+1 for x in np.arange(len(draw_points_l1))]
        plt.plot(x, draw_points_l1)
        plt.plot(x, draw_points_l2)
        plt.plot(x, draw_points_l3)
        plt.plot(x, draw_points_l4)
        plt.legend(['l1', 'l2', 'l3', 'total loss'], loc='lower left')
        plt.xlabel('epoches')
        plt.ylabel('losses')
        plt.show()