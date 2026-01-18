"""
一键下载并生成所有 SAITS 数据集

此脚本等效于依次运行:
1. data_downloading.sh - 下载原始数据集
2. dataset_generating.sh - 生成预处理数据集

适用于 Windows/Linux/Mac 全平台

作者: GitHub Copilot
日期: 2026-01-16
许可: MIT
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path


class DatasetDownloadAndGenerator:
    """数据集下载和生成器"""
    
    def __init__(self, base_dir=None):
        """
        初始化
        
        Args:
            base_dir: 基础目录,默认为脚本所在目录
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        
        self.raw_data_dir = self.base_dir / "RawData"
        self.generated_datasets_dir = self.base_dir.parent / "generated_datasets"
        
    def print_step(self, step_num, total_steps, message):
        """打印步骤信息"""
        print(f"\n{'='*70}")
        print(f"[步骤 {step_num}/{total_steps}] {message}")
        print(f"{'='*70}")
        
    def download_file(self, url, save_path, description=""):
        """
        下载文件(带进度显示)
        
        Args:
            url: 下载链接
            save_path: 保存路径
            description: 文件描述
        """
        print(f"📥 正在下载: {description or save_path.name}")
        print(f"   URL: {url}")
        
        try:
            # 使用 urllib 下载(带进度)
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\r   进度: {percent}% ")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, save_path, reporthook)
            print(f"\n✅ 下载完成: {save_path.name}")
            return True
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            return False
    
    def extract_tar_gz(self, tar_path, extract_to):
        """解压 .tar.gz 文件"""
        print(f"📦 正在解压: {tar_path.name}")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_to)
            print(f"✅ 解压完成")
            return True
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to):
        """解压 .zip 文件"""
        print(f"📦 正在解压: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✅ 解压完成")
            return True
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False
    
    def download_physionet2012(self):
        """下载 PhysioNet-2012 数据集"""
        self.print_step(1, 8, "下载 PhysioNet-2012 数据集")
        
        physio_dir = self.raw_data_dir / "Physio2012_mega"
        physio_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载文件列表
        files_to_download = [
            ("https://www.physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download", "set-a.tar.gz"),
            ("https://www.physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download", "set-b.tar.gz"),
            ("https://www.physionet.org/files/challenge-2012/1.0.0/set-c.tar.gz?download", "set-c.tar.gz"),
            ("https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt?download", "Outcomes-a.txt"),
            ("https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-b.txt?download", "Outcomes-b.txt"),
            ("https://www.physionet.org/files/challenge-2012/1.0.0/Outcomes-c.txt?download", "Outcomes-c.txt"),
        ]
        
        # 下载所有文件
        for url, filename in files_to_download:
            save_path = physio_dir / filename
            if save_path.exists():
                print(f"⏭️  跳过已存在的文件: {filename}")
                continue
            self.download_file(url, save_path, f"PhysioNet-2012 {filename}")
        
        # 解压并合并
        print(f"\n📂 正在合并数据集...")
        mega_dir = physio_dir / "mega"
        mega_dir.mkdir(exist_ok=True)
        
        for tar_name in ["set-a.tar.gz", "set-b.tar.gz", "set-c.tar.gz"]:
            tar_path = physio_dir / tar_name
            if tar_path.exists():
                self.extract_tar_gz(tar_path, physio_dir)
                
                # 移动文件到 mega 目录
                set_dir = physio_dir / tar_name.replace(".tar.gz", "")
                if set_dir.exists():
                    for file in set_dir.iterdir():
                        shutil.move(str(file), str(mega_dir))
                    set_dir.rmdir()
        
        print(f"✅ PhysioNet-2012 数据集准备完成")
    
    def download_air_quality(self):
        """下载 Air Quality 数据集"""
        self.print_step(2, 8, "下载 Air Quality 数据集")
        
        air_dir = self.raw_data_dir / "AirQuality"
        air_dir.mkdir(parents=True, exist_ok=True)
        
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"
        zip_path = air_dir / "PRSA2017_Data_20130301-20170228.zip"
        
        if not zip_path.exists():
            self.download_file(url, zip_path, "Air Quality 数据集")
        else:
            print(f"⏭️  跳过已存在的文件: {zip_path.name}")
        
        if zip_path.exists() and not (air_dir / "PRSA_Data_20130301-20170228").exists():
            self.extract_zip(zip_path, air_dir)
        
        print(f"✅ Air Quality 数据集准备完成")
    
    def download_electricity(self):
        """下载 Electricity 数据集"""
        self.print_step(3, 8, "下载 Electricity 数据集")
        
        elec_dir = self.raw_data_dir / "Electricity"
        elec_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
        zip_path = elec_dir / "LD2011_2014.txt.zip"
        
        if not zip_path.exists():
            self.download_file(url, zip_path, "Electricity 数据集")
        else:
            print(f"⏭️  跳过已存在的文件: {zip_path.name}")
        
        if zip_path.exists() and not (elec_dir / "LD2011_2014.txt").exists():
            self.extract_zip(zip_path, elec_dir)
        
        print(f"✅ Electricity 数据集准备完成")
    
    def download_ett(self):
        """下载 ETT 数据集"""
        self.print_step(4, 8, "下载 ETT 数据集")
        
        ett_dir = self.raw_data_dir / "ETT"
        ett_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"
        csv_path = ett_dir / "ETTm1.csv"
        
        if not csv_path.exists():
            self.download_file(url, csv_path, "ETTm1.csv")
        else:
            print(f"⏭️  跳过已存在的文件: {csv_path.name}")
        
        print(f"✅ ETT 数据集准备完成")
    
    def run_python_script(self, script_name, args):
        """
        运行 Python 脚本
        
        Args:
            script_name: 脚本名称
            args: 命令行参数列表
        """
        script_path = self.base_dir / script_name
        
        if not script_path.exists():
            print(f"❌ 脚本不存在: {script_path}")
            return False
        
        print(f"🚀 正在运行: {script_name}")
        print(f"   参数: {' '.join(args)}")
        
        try:
            # 构建完整命令
            cmd = [sys.executable, str(script_path)] + args
            
            # 运行脚本
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )
            
            # 显示输出
            if result.stdout:
                print(result.stdout)
            
            if result.returncode == 0:
                print(f"✅ {script_name} 运行成功")
                return True
            else:
                print(f"❌ {script_name} 运行失败")
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 运行失败: {e}")
            return False
    
    def generate_physionet2012(self):
        """生成 PhysioNet-2012 数据集"""
        self.print_step(5, 8, "生成 PhysioNet-2012 预处理数据集")
        
        args = [
            "--raw_data_path", str(self.raw_data_dir / "Physio2012_mega" / "mega"),
            "--outcome_files_dir", str(self.raw_data_dir / "Physio2012_mega"),
            "--dataset_name", "physio2012_37feats_01masked",
            "--saving_path", str(self.generated_datasets_dir)
        ]
        
        return self.run_python_script("gene_PhysioNet2012_dataset.py", args)
    
    def generate_air_quality(self):
        """生成 Air Quality 数据集"""
        self.print_step(6, 8, "生成 Air Quality 预处理数据集")
        
        args = [
            "--file_path", str(self.raw_data_dir / "AirQuality" / "PRSA_Data_20130301-20170228"),
            "--seq_len", "24",
            "--artificial_missing_rate", "0.1",
            "--dataset_name", "AirQuality_seqlen24_01masked",
            "--saving_path", str(self.generated_datasets_dir)
        ]
        
        return self.run_python_script("gene_UCI_BeijingAirQuality_dataset.py", args)
    
    def generate_electricity(self):
        """生成 Electricity 数据集"""
        self.print_step(7, 8, "生成 Electricity 预处理数据集")
        
        args = [
            "--file_path", str(self.raw_data_dir / "Electricity" / "LD2011_2014.txt"),
            "--artificial_missing_rate", "0.1",
            "--seq_len", "100",
            "--dataset_name", "Electricity_seqlen100_01masked",
            "--saving_path", str(self.generated_datasets_dir)
        ]
        
        return self.run_python_script("gene_UCI_electricity_dataset.py", args)
    
    def generate_ett(self):
        """生成 ETT 数据集"""
        self.print_step(8, 8, "生成 ETT 预处理数据集")
        
        args = [
            "--file_path", str(self.raw_data_dir / "ETT" / "ETTm1.csv"),
            "--artificial_missing_rate", "0.1",
            "--seq_len", "24",
            "--sliding_len", "12",
            "--dataset_name", "ETTm1_seqlen24_01masked",
            "--saving_path", str(self.generated_datasets_dir)
        ]
        
        return self.run_python_script("gene_ETTm1_dataset.py", args)
    
    def run_all(self):
        """运行所有步骤"""
        print("\n" + "="*70)
        print("🎯 开始执行数据集下载和生成流程")
        print("="*70)
        
        # 阶段 1: 下载原始数据
        print("\n" + "🔽"*35)
        print("阶段 1: 下载原始数据集")
        print("🔽"*35)
        
        self.download_physionet2012()
        self.download_air_quality()
        self.download_electricity()
        self.download_ett()
        
        # 阶段 2: 生成预处理数据
        print("\n" + "⚙️"*35)
        print("阶段 2: 生成预处理数据集")
        print("⚙️"*35)
        
        self.generate_physionet2012()
        self.generate_air_quality()
        self.generate_electricity()
        self.generate_ett()
        
        # 完成
        print("\n" + "="*70)
        print("🎉 所有数据集下载和生成完成!")
        print("="*70)
        print(f"\n📁 原始数据位置: {self.raw_data_dir}")
        print(f"📁 预处理数据位置: {self.generated_datasets_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="一键下载并生成所有 SAITS 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载并生成所有数据集
  python download_and_generate_all_datasets.py
  
  # 仅下载原始数据
  python download_and_generate_all_datasets.py --download-only
  
  # 仅生成预处理数据(假设已下载原始数据)
  python download_and_generate_all_datasets.py --generate-only
        """
    )
    
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="仅下载原始数据,不生成预处理数据集"
    )
    
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="仅生成预处理数据集,假设原始数据已下载"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="指定基础目录(默认为脚本所在目录)"
    )
    
    args = parser.parse_args()
    
    # 创建生成器实例
    generator = DatasetDownloadAndGenerator(base_dir=args.base_dir)
    
    # 根据参数执行不同操作
    if args.download_only:
        print("\n🔽 仅执行下载操作\n")
        generator.download_physionet2012()
        generator.download_air_quality()
        generator.download_electricity()
        generator.download_ett()
        print("\n✅ 下载完成!")
        
    elif args.generate_only:
        print("\n⚙️ 仅执行生成操作\n")
        generator.generate_physionet2012()
        generator.generate_air_quality()
        generator.generate_electricity()
        generator.generate_ett()
        print("\n✅ 生成完成!")
        
    else:
        # 执行完整流程
        generator.run_all()


if __name__ == "__main__":
    main()
