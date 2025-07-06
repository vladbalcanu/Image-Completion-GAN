# flake8: noqa
import torch
import torch.nn as nn
import torch.optim as optim
import os
from networks.discriminator_network import Discriminator
from networks.completion_generator_network import ImageCompletionGenerator
from losses.adversarial_loss import AdversarialLoss
from losses.style_loss import StyleLoss
from losses.perceptual_loss import PerceptualLoss

class CompletionModel(nn.Module):
    def __init__(self):
        super(CompletionModel, self).__init__()
        self.name = "ImageCompletionModel"
        self.iteration = 0
        self.generator_optimizer_learning_rate = 0.000001
        self.discriminator_optimizer_learning_rate = 0.0000001
        self.generator_beta1 = 0.0
        self.generator_beta2 = 0.9
        self.discriminator_beta1 = 0.0
        self.discriminator_beta2 = 0.9
        self.adversial_loss_weight = 0.1
        self.perceptual_loss_weight = 0.1
        self.relative_l1_loss_weight = 1.0
        self.style_loss_weight = 250

        base_path = os.path.dirname(os.path.abspath(__file__))
        self.generator_weights_path = os.path.join(base_path, "completion_model_weights", self.name + '_gen')
        self.discriminator_weights_path = os.path.join(base_path, "completion_model_weights", self.name + '_dis')

        generator = ImageCompletionGenerator()
        discriminator = Discriminator(3)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss()

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.generator_adam_optimizer = optim.Adam(params=generator.parameters(),
                                        lr=float(self.generator_optimizer_learning_rate),
                                        betas=(self.generator_beta1, self.generator_beta2))

        self.discriminator_adam_optimizer = optim.Adam(params=discriminator.parameters(),
                                        lr=float(self.discriminator_optimizer_learning_rate),
                                        betas=(self.discriminator_beta1, self.discriminator_beta2))

    def process(self, images, edges, masks):
        self.generator_adam_optimizer.zero_grad()
        self.discriminator_adam_optimizer.zero_grad()
        generator_total_loss = 0
        discriminator_loss = 0

        forward_result = self(images, edges, masks)

        discriminator_real_input = images
        discriminator_generated_input = forward_result.detach()
        discriminator_real_result, _ = self.discriminator(discriminator_real_input)
        discriminator_generated_result, _ = self.discriminator(discriminator_generated_input)
        discriminator_real_loss = self.adversarial_loss(discriminator_real_result, True)
        discriminator_generated_loss = self.adversarial_loss(discriminator_generated_result, False)
        discriminator_loss += (discriminator_real_loss + discriminator_generated_loss) / 2

        generator_result = forward_result
        generator_generated_result, _ = self.discriminator(generator_result)
        generator_adv_loss = self.adversarial_loss(generator_generated_result, True) * self.adversial_loss_weight
        generator_total_loss += generator_adv_loss

        generator_l1_loss = self.l1_loss(forward_result, images) * self.relative_l1_loss_weight / torch.mean(masks)
        generator_total_loss += generator_l1_loss

        generator_perceptual_loss = self.perceptual_loss(forward_result, images)
        generator_perceptual_loss = generator_perceptual_loss * self.perceptual_loss_weight
        generator_total_loss += generator_perceptual_loss

        generator_style_loss = self.style_loss(forward_result * masks, images * masks)
        generator_style_loss = generator_style_loss * self.style_loss_weight
        generator_total_loss += generator_style_loss

        loss_logs = [
            ("discriminator2_loss", discriminator_loss.item()),
            ("generator2_loss", generator_adv_loss.item()),
            ("l1_loss", generator_l1_loss.item()),
            ("perceptual_loss", generator_perceptual_loss.item()),
            ("style_loss", generator_style_loss.item()),
        ]
        self.iteration += 1

        return forward_result, generator_total_loss, discriminator_loss, loss_logs

    def forward(self, images, edges, masks):
        masked_images = (images * (1 - masks).float()) + masks
        generator_inputs = torch.cat((masked_images, edges, masks), dim=1)
        generator_result = self.generator(generator_inputs)
        return generator_result

    def backward(self, generator_total_loss, discriminator_loss):
        discriminator_loss.backward(retain_graph=True)
        generator_total_loss.backward(retain_graph=True)
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
