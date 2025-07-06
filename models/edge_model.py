# flake8: noqa
import torch
import torch.nn as nn
import os
from networks.discriminator_network import Discriminator
from networks.edge_generator_network import EdgeGenerator
from losses.adversarial_loss import AdversarialLoss
from torch.optim import Adam

class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.name = "EdgeModel"
        self.generator_optimizer_learning_rate = 0.000001
        self.discriminator_optimizer_learning_rate = 0.0000001
        self.generator_beta1 = 0.0
        self.generator_beta2 = 0.9
        self.discriminator_beta1 = 0.0
        self.discriminator_beta2 = 0.9
        self.feature_matching_loss_weight = 10.0
        self.iteration = 0

        base_path = os.path.dirname(os.path.abspath(__file__))
        self.generator_weights_path = os.path.join(base_path, "edge_model_weights", self.name + '_gen')
        self.discriminator_weights_path = os.path.join(base_path, "edge_model_weights", self.name + '_dis')

        generator = EdgeGenerator()
        discriminator = Discriminator(2)
        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.generator_adam_optimizer = Adam(params=generator.parameters(),
                                             lr=float(self.generator_optimizer_learning_rate),
                                             betas=(self.generator_beta1, self.generator_beta2))
        self.discriminator_adam_optimizer = Adam(params=discriminator.parameters(),
                                                 lr=float(self.discriminator_optimizer_learning_rate),
                                                 betas=(self.discriminator_beta1, self.discriminator_beta2))

    def process(self, grayscale_images, edges, masks):
        self.generator_adam_optimizer.zero_grad()
        self.discriminator_adam_optimizer.zero_grad()
        generator_total_loss = 0
        discriminator_total_loss = 0
        
        forward_result = self(grayscale_images, edges, masks)

        discriminator_real_input = torch.cat((grayscale_images, edges), dim=1)
        discriminator_generated_input = torch.cat((grayscale_images, forward_result.detach()), dim=1) # se utilizează detach pentru a nu fi folosit in backpropagation
        discriminator_real_result, discriminator_real_features = self.discriminator(discriminator_real_input)
        discriminator_generated_result, _ = self.discriminator(discriminator_generated_input)
        discriminator_real_adv_loss = self.adversarial_loss(discriminator_real_result, True)
        discriminator_generated_adv_loss = self.adversarial_loss(discriminator_generated_result, False)
        discriminator_total_loss += (discriminator_real_adv_loss + discriminator_generated_adv_loss) / 2

        generated_input = torch.cat((grayscale_images, forward_result), dim=1)
        generator_generated_result, generator_generated_features = self.discriminator(generated_input)
        generator_adv_loss = self.adversarial_loss(generator_generated_result, True)
        generator_total_loss += generator_adv_loss
        feature_matching_loss = 0
        for i in range(len(discriminator_real_features)):
            feature_matching_loss += self.l1_loss(generator_generated_features[i], discriminator_real_features[i].detach())
        feature_matching_loss = feature_matching_loss * self.feature_matching_loss_weight
        generator_total_loss += feature_matching_loss

        loss_logs = [
            ("discriminator1_loss", discriminator_total_loss.item()),
            ("generator1_loss", generator_adv_loss.item()),
            ("feature_matching_loss", feature_matching_loss.item()),
        ]
        self.iteration = self.iteration + 1

        return forward_result, generator_total_loss, discriminator_total_loss, loss_logs

    def forward(self, grayscale_images, edges, masks):
        edges_masked = (edges * (1 - masks))
        grayscale_images_masked = (grayscale_images * (1 - masks)) + masks
        generator_inputs = torch.cat((grayscale_images_masked, edges_masked, masks), dim=1)
        generator_result = self.generator(generator_inputs)
        return generator_result

    def backward(self, generator_loss, discriminator_loss):
        discriminator_loss.backward()
        generator_loss.backward()
        self.discriminator_adam_optimizer.step()
        self.generator_adam_optimizer.step()

    def load(self, suffix: str = None):
        if suffix != None:
            generator_save_path = self.generator_weights_path + '_' + suffix + '.pth'
            discriminator_save_path = self.discriminator_weights_path + '_' + suffix + '.pth'
        else:
            generator_save_path = self.generator_weights_path + '.pth'
            discriminator_save_path = self.discriminator_weights_path + '.pth'

        if os.path.exists(generator_save_path) and os.path.exists(discriminator_save_path):
            print('Loading generator and discriminator weights for %s\n' % self.name)
            generator_data = torch.load(generator_save_path)
            discriminator_data = torch.load(discriminator_save_path)
            self.generator.load_state_dict(generator_data['generator'])
            self.discriminator.load_state_dict(discriminator_data['discriminator'])
            self.iteration = generator_data['iteration']
        else:
            if not os.path.exists(generator_save_path):
                print('Generator weights could not be found!\n')
            if not os.path.exists(discriminator_save_path):
                print('Discriminator weights could not be found!\n')
            print("No weights loaded for %s\n", self.name)
            

    def save(self, suffix: str = None):
        if suffix != None:
            generator_save_path = self.generator_weights_path + '_' + suffix + '.pth'
            discriminator_save_path = self.discriminator_weights_path + '_' + suffix + '.pth'
        else:
            generator_save_path = self.generator_weights_path + '.pth'
            discriminator_save_path = self.discriminator_weights_path + '.pth'

        print('Saving generator and discriminator weights for %s\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, generator_save_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, discriminator_save_path)
