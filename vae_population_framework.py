import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Çalışmanın ilk bölümünde başarılı olan metotlar
SUCCESSFUL_METHODS = ['fifo_heuristic', 'most_work_remaining']

class JSPSolutionDataset(Dataset):
    
    def __init__(self, solutions: np.ndarray, fitnesses: np.ndarray, elite_ratio: float = 0.9):

        self.solutions = np.array(solutions, dtype=np.float32)
        self.fitnesses = np.array(fitnesses, dtype=np.float32)
        
        # Sadece geçerli çözümleri al
        feasible_mask = self.fitnesses < 1e6
        self.solutions = self.solutions[feasible_mask]
        self.fitnesses = self.fitnesses[feasible_mask]
        
        if len(self.solutions) == 0:
            raise ValueError("Geçerli çözüm bulunamadı!")
        
        # Çok az veri varsa tümünü elite olarak kabul et
        if len(self.solutions) <= 3:
            elite_ratio = 1.0
        
        n_elite = max(1, int(len(self.solutions) * elite_ratio))
        elite_indices = np.argsort(self.fitnesses)[:n_elite]
        
        self.elite_solutions = self.solutions[elite_indices]
        self.elite_fitnesses = self.fitnesses[elite_indices]
    
    def __len__(self):
        return len(self.elite_solutions)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.elite_solutions[idx])

class CompactVAE(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int = None):
        super(CompactVAE, self).__init__()
        
        self.input_dim = input_dim
        
        # Otomatik latent boyut hesaplama
        if latent_dim is None:
            self.latent_dim = max(4, min(16, input_dim // 20))
        else:
            self.latent_dim = latent_dim
        
        # Küçük ve basit mimari
        hidden_dim = max(32, min(128, input_dim // 4))
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dim // 2, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
    
    def generate_samples(self, num_samples: int):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            return self.decode(z).numpy()

class ImprovedVAETrainer:
    
    def __init__(self, model: CompactVAE, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Verbose parametresini kaldırdık
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=15
        )
        
        self.losses = []
    
    def vae_loss(self, recon_x, x, mu, log_var, beta=0.1):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    def train(self, dataset: JSPSolutionDataset, num_epochs: int = None, batch_size: int = None):
        
        # Otomatik parametre ayarlama
        if num_epochs is None:
            num_epochs = max(100, min(500, 1000 // len(dataset)))
        
        if batch_size is None:
            batch_size = max(1, min(8, len(dataset)))
        
        # Çok küçük dataset için batch_size = 1
        if len(dataset) <= 2:
            batch_size = 1
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            
            for batch_data in dataloader:
                self.optimizer.zero_grad()
                
                recon_batch, mu, log_var = self.model(batch_data)
                loss, recon_loss, kl_loss = self.vae_loss(recon_batch, batch_data, mu, log_var)
                
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.losses.append(avg_loss)
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)
            
            # Erken durma
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > 50:  
                print(f"  Early stopping at epoch {epoch}")
                break
            
            # Progress
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                print(f"  Epoch {epoch:3d}: Loss={avg_loss:.4f}, "
                      f"Recon={epoch_recon/len(dataloader):.4f}, "
                      f"KL={epoch_kl/len(dataloader):.6f}")
        
    
    def save_model(self, filepath: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim
            },
            'losses': self.losses
        }, filepath)

class ImprovedDataCollector:
    
    def __init__(self):
        self.instances_info = {
            'ta02': (15, 15, 225),
            'ta12': (20, 15, 300),
            'ta22': (20, 20, 400),
            'ta32': (30, 15, 450),
            'ta42': (30, 20, 600),
            'ta52': (50, 15, 750)
        }
    
    def collect_solutions_for_instance(self, instance: str, target_count: int = 50):
        
        # Problem instance'ını güncelle
        from jssp_framework import update_problem_instance, PopulationInitializer
        from jssp_framework import num_jobs, num_machines, machine_sequence, processing_times, cached_decode, evaluate_jssp_solution
        
        update_problem_instance(instance)
        
        # Global değişkenleri yeniden import et
        import jssp_framework
        dim = jssp_framework.num_jobs * jssp_framework.num_machines
        
        solutions = []
        fitnesses = []
        
        # Sadece başarılı metotları kullan
        lb, ub = np.zeros(dim), np.ones(dim)
        initializer = PopulationInitializer(dim, 80, lb, ub)  # Daha büyük popülasyon
        
        # Başarılı metotlarla daha fazla çalışma
        successful_methods = SUCCESSFUL_METHODS
        runs_per_method = max(10, target_count // (len(successful_methods) * 20))
        
        for method_name in successful_methods:
 
            print(f"   {method_name} ile {runs_per_method} çalışma...")
            method = getattr(initializer, method_name)
            
            method_solutions = 0
            for run in range(runs_per_method):
                try:
                    population = method()
                    
                    # Tüm popülasyonu değerlendir
                    for individual in population:
                        try:
                            schedule, _ = jssp_framework.cached_decode(tuple(individual))
                            fitness = jssp_framework.evaluate_jssp_solution(
                                schedule, 
                                jssp_framework.machine_sequence, 
                                jssp_framework.processing_times
                            )
                            
                            if fitness < 1e6:  # Geçerli çözüm
                                solutions.append(individual.copy())
                                fitnesses.append(float(fitness))
                                method_solutions += 1
                                
                        except Exception:
                            continue
                
                except Exception as e:
                    print(f"  Run {run} failed: {e}")
                    continue
                
                # Erken çıkış
                if len(solutions) >= target_count:
                    break
            
            print(f"  {method_name}: {method_solutions} geçerli çözüm")
            
            if len(solutions) >= target_count:
                break
        
        # Eğer hala yetersizse random deneme
        if len(solutions) < min(10, target_count // 5):
            print(f"  Random çözümlerle tamamlama...")
            
            for _ in range(100):  
                random_solution = np.random.uniform(0, 1, dim)
                try:
                    schedule, _ = jssp_framework.cached_decode(tuple(random_solution))
                    fitness = jssp_framework.evaluate_jssp_solution(
                        schedule, jssp_framework.machine_sequence, jssp_framework.processing_times
                    )
                    
                    if fitness < 1e6:
                        solutions.append(random_solution.copy())
                        fitnesses.append(float(fitness))
                        
                        if len(solutions) >= target_count:
                            break
                except:
                    continue
        
        print(f"  Toplam: {len(solutions)} geçerli çözüm")
        if len(solutions) > 0:
            print(f"     En iyi fitness: {min(fitnesses):.2f}")
            print(f"     Ortalama fitness: {np.mean(fitnesses):.2f}")
        
        return solutions, fitnesses

class MultiDimensionalVAESystem:
    """Basitleştirilmiş Multi-Dimensional VAE sistemi"""
    
    def __init__(self):
        self.vae_models = {}  # {dimension: model}
        self.collectors = ImprovedDataCollector()
        
    def train_all_dimensions(self, min_solutions_per_dim: int = 15):
        """Tüm boyutlar için VAE eğitimi"""
        print(" Multi-dimensional VAE eğitimi")
        
        successful_dims = []
        
        for instance, (jobs, machines, dim) in self.collectors.instances_info.items():
            
            # Veri toplama
            solutions, fitnesses = self.collectors.collect_solutions_for_instance(
                instance, target_count=min_solutions_per_dim * 3
            )
            
            if len(solutions) < min_solutions_per_dim:
                print(f" {instance}: Yetersiz veri ({len(solutions)} < {min_solutions_per_dim})")
                continue
            
            try:
                # Dataset oluştur
                solutions_array = np.array(solutions, dtype=np.float32)
                fitnesses_array = np.array(fitnesses, dtype=np.float32)
                
                dataset = JSPSolutionDataset(solutions_array, fitnesses_array, elite_ratio=0.8)
                
                # Model oluştur ve eğit
                model = CompactVAE(input_dim=dim)
                trainer = ImprovedVAETrainer(model)
                
                trainer.train(dataset)
                
                # Modeli sakla
                model_filename = f'compact_vae_{dim}d.pth'
                trainer.save_model(model_filename)
                
                self.vae_models[dim] = model
                successful_dims.append(dim)
                
                print(f" {instance} için VAE eğitimi başarılı!")
                
            except Exception as e:
                print(f" {instance} eğitimi başarısız: {e}")
                continue
        
        # Sonuçları kaydet
        model_info = {
            'successful_dimensions': successful_dims,
            'total_models': len(successful_dims),
            'model_files': {dim: f'compact_vae_{dim}d.pth' for dim in successful_dims}
        }
        
        with open('multi_vae_models.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\n Eğitim tamamlandı! {len(successful_dims)} boyut için VAE modeli hazır:")
        for dim in successful_dims:
            print(f"   {dim}D model")
        
        return successful_dims
    
    def load_models(self):
        """Eğitilmiş modelleri yükle"""
        try:
            with open('multi_vae_models.json', 'r') as f:
                model_info = json.load(f)
            
            loaded_count = 0
            for dim_str in model_info['successful_dimensions']:
                dim = int(dim_str)
                model_file = f'compact_vae_{dim}d.pth'
                
                if os.path.exists(model_file):
                    model = CompactVAE(input_dim=dim)
                    checkpoint = torch.load(model_file, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    self.vae_models[dim] = model
                    loaded_count += 1
            
            print(f" {loaded_count} VAE modeli yüklendi")
            return loaded_count > 0
            
        except FileNotFoundError:
            print(" Model dosyaları bulunamadı")
            return False
    
    def generate_population_for_dimension(self, dim: int, pop_size: int):
        """Belirli boyut için popülasyon üret"""
        if dim not in self.vae_models:
            raise ValueError(f"Boyut {dim} için VAE modeli yok!")
        
        model = self.vae_models[dim]
        population = model.generate_samples(pop_size)
        
        # [0,1] aralığında sınırla
        population = np.clip(population, 0, 1)
        
        return population
    
    def test_generation(self):
        """Tüm boyutlarda üretimi test et"""
        print(" VAE üretim testi...")
        
        for dim, model in self.vae_models.items():
            print(f"\n  {dim}D test:")
            
            # Küçük popülasyon üret
            population = self.generate_population_for_dimension(dim, 10)
            
            print(f"    Üretilen: {population.shape}")
            print(f"    Min/Max: {population.min():.3f}/{population.max():.3f}")
            print(f"    Geçerlilik: {np.all((population >= 0) & (population <= 1))}")

# Ana çalıştırma fonksiyonu
def main():
    """Ana VAE sistemi"""
    
    system = MultiDimensionalVAESystem()
    
    print("Seçenekler:")
    print("1. VAE modellerini eğit")
    print("2. Modelleri yükle ve test et")
    print("3. Tam pipeline çalıştır")
    
    choice = input("Seçiminiz (1-3): ").strip()
    
    if choice == '1':
        successful_dims = system.train_all_dimensions(min_solutions_per_dim=10)
        if successful_dims:
            print(" Eğitim başarılı!")
        else:
            print(" Hiçbir model eğitilemedi!")
    
    elif choice == '2':
        if system.load_models():
            system.test_generation()
        else:
            print(" Model yüklenemedi!")
    
    elif choice == '3':
        print(" Tam pipeline çalışıyor...")
        
        # Eğitim
        successful_dims = system.train_all_dimensions(min_solutions_per_dim=8)
        
        if successful_dims:
            # Test
            system.test_generation()
            print(" Pipeline tamamlandı!")
        else:
            print(" Pipeline başarısız!")
    
    else:
        print("Geçersiz seçim, eğitim başlatılıyor...")
        system.train_all_dimensions()

if __name__ == "__main__":
    main()