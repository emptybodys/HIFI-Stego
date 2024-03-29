import torch 
from dataset import VC_Dataset
from torch.utils.data import Dataset, DataLoader
from hparams import hparams
from model import GeneratorLeft, GeneratorRight, SteganographyAlgorithm, AfterProcess
import os
import logging
import torch.nn as nn
import difflib

def adjust_lr_rate(optimizer,lr,lr_decay):
    lr_new = max(0., lr - lr_decay)
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr_new
    return lr_new,optimizer
    

if __name__ == "__main__":
    
    # 定义log文件
    file_log = "starGan.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(file_log),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()   
    
    # 定义device
    device = torch.device("cuda:0")
    
    # 获取模型参数
    para = hparams()
    
    # 隐写算法实例化
    steganography = SteganographyAlgorithm().to(device)
    
    # 模型实例化
    m_GeneratorLeft = GeneratorLeft(para.n_spk).to(device)
    
    m_GeneratorRight = GeneratorRight(para.n_spk).to(device)
    
    m_AfterChunk = AfterProcess().to(device)
    
    # 定义优化器
    g_l_lr =  para.g_lr
    g_l_optimizer = torch.optim.Adam(m_GeneratorLeft.parameters(), g_l_lr, [0.5, 0.999])
    
    g_r_lr =  para.g_lr
    g_r_optimizer = torch.optim.Adam(m_GeneratorRight.parameters(), g_r_lr, [0.5, 0.999])
    
    p_lr =  para.p_lr
    p_optimizer = torch.optim.Adam(m_AfterChunk.parameters(), p_lr, [0.5, 0.999])
    
    # 损失函数
    CELoss = nn.MSELoss()
    
    # 定义数据集
    m_Dataset= VC_Dataset(para)
    m_DataLoader = DataLoader(m_Dataset,batch_size = 1,shuffle = True, num_workers = 1)
    
    n_step = 0
    
    # 层
    layer = 1
    
    for epoch in range(para.n_epoch):
        # 打乱顺序
        m_Dataset.gen_random_pair()
        
        loss_g_1 = 0.0
        loss_g_2 = 0.0
        diff = 0.0
        
        for i, sample_batch in enumerate(m_DataLoader):
            n_step = n_step+1

            # 读取数据
            real_A = sample_batch[0]
            one_hot_A = sample_batch[1]
            
            real_A = real_A.to(device).float()
            one_hot_A = one_hot_A.to(device).float()
            
            # 无隐写状态
            latent_space = m_GeneratorLeft(real_A)
            fake_A = m_GeneratorRight(latent_space, one_hot_A, False)
            h, w = m_GeneratorRight.get_write_length(layer)

            loss_g_r_recovery_audio = CELoss(fake_A, real_A)
            # 每一步都应该相同
            loss_g_r_0 = CELoss(m_GeneratorLeft.get_parameter(0), m_GeneratorRight.get_parameter(5))
            loss_g_r_1 = CELoss(m_GeneratorLeft.get_parameter(1), m_GeneratorRight.get_parameter(4))
            loss_g_r_2 = CELoss(m_GeneratorLeft.get_parameter(2), m_GeneratorRight.get_parameter(3))
            loss_g_r_3 = CELoss(m_GeneratorLeft.get_parameter(3), m_GeneratorRight.get_parameter(2))
            loss_g_r_4 = CELoss(m_GeneratorLeft.get_parameter(4), m_GeneratorRight.get_parameter(1))
 
            loss_g_1 = loss_g_r_recovery_audio + loss_g_r_1 + \
                                               loss_g_r_2 + \
                                               loss_g_r_3 + \
                                               loss_g_r_4 + \
                                               loss_g_r_0    
            
            g_l_optimizer.zero_grad()
            g_r_optimizer.zero_grad()
            
            loss_g_1.backward()
            g_l_optimizer.step()
            g_r_optimizer.step()

            # 有隐写状态
            secret_inf = steganography.write_secret(h, w).to(device)
            latent_space = m_GeneratorLeft(real_A)
            fake_A_2 = m_GeneratorRight(latent_space, one_hot_A, True, secret_inf, layer)
            
            loss_g_r_secret_undetect = CELoss(real_A, fake_A_2)
            
            # 有隐写分层
            fake_A_2_latent = m_GeneratorLeft(fake_A_2)
            fake_A_2_1 = m_GeneratorLeft.get_parameter(6-layer).to(device)

            # 分离层
            chunk_secret = m_AfterChunk(fake_A_2_1, layer)     
            loss_g_l_secret_recovery_1 = CELoss(chunk_secret, secret_inf)
            
            # 恢复的子串对比
            sec_rev = steganography.inverse(chunk_secret)
            diff = difflib.SequenceMatcher(None, sec_rev, steganography.secretinformation[:steganography.get_realusedlength()]).quick_ratio()
            loss_r_p_secret = 1.0 - torch.tensor(diff)
            
            loss_g_2 = loss_g_r_secret_undetect + loss_g_l_secret_recovery_1 * 10 + loss_r_p_secret * 10
            
            g_l_optimizer.zero_grad()
            g_r_optimizer.zero_grad()
            p_optimizer.zero_grad()
            
            loss_g_2.backward()
            g_l_optimizer.step()
            g_r_optimizer.step()
            p_optimizer.step()
                  
            # 调整lr
            if n_step>para.start_decay and n_step%(para.lr_update_step)==0:

               g_l_lr, g_l_optimizer= adjust_lr_rate(g_l_optimizer,g_l_lr,para.decay_g_l)
               g_r_lr, g_r_optimizer= adjust_lr_rate(g_r_optimizer,g_r_lr,para.decay_g_r)
               p_lr, p_optimizer =  adjust_lr_rate(g_r_optimizer,p_lr,para.decay_p)
            
           #  模型保存            
            if  n_step %(para.save_step) ==0:
                path_save = os.path.join(para.path_save,str(n_step))
                os.makedirs(path_save,exist_ok=True)
                
                torch.save({'model_G_L':m_GeneratorLeft.state_dict(),
                            'model_G_R':m_GeneratorRight.state_dict(),
                            'model_A_P':m_AfterChunk.state_dict(),
                            'opt_G_L':g_l_optimizer.state_dict(),
                            'opt_G_R':g_r_optimizer.state_dict()},
                            os.path.join(path_save,'model.pick'))
                            
        # 打印log
        logger.info("epoch %d g_1_loss= %f,g_2_loss= %f,s_r_loss= %f"%(epoch,loss_g_1,loss_g_2,diff))    
    
    torch.save({'model_G_L':m_GeneratorLeft.state_dict(),
                'model_G_R':m_GeneratorRight.state_dict(),
                'model_A_P':m_AfterChunk.state_dict(),
                'opt_G_L':g_l_optimizer.state_dict(),
                'opt_G_R':g_r_optimizer.state_dict()},
                os.path.join(path_save,'final_model.pick'))
               
               
               
               
               
              
            

            
            
            
                
            
            
            
            
        
    
    
    
    
    
