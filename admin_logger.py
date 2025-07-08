# admin_logger.py
# comprehensive logging system for admin access to all user data and results
# tracks uploads, parameters, results, and user activity

import os
import json
import datetime
import pandas as pd
import hashlib
import shutil
from pathlib import Path

class AdminLogger:
    def __init__(self, log_dir="admin_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # create subdirectories
        (self.log_dir / "uploads").mkdir(exist_ok=True)
        (self.log_dir / "results").mkdir(exist_ok=True)
        (self.log_dir / "parameters").mkdir(exist_ok=True)
        (self.log_dir / "processed_images").mkdir(exist_ok=True)
        
        self.session_log_file = self.log_dir / "session_log.jsonl"
        self.summary_file = self.log_dir / "admin_summary.csv"
    
    def generate_session_id(self, user_identifier=None):
        # generate unique session id
        timestamp = datetime.datetime.now().isoformat()
        if user_identifier:
            session_data = f"{user_identifier}_{timestamp}"
        else:
            session_data = f"anonymous_{timestamp}"
        
        return hashlib.md5(session_data.encode()).hexdigest()[:12]
    
    def log_session(self, session_id, user_info):
        # log session start
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "action": "session_start",
            "user_info": user_info
        }
        
        self._append_log(log_entry)
    
    def log_upload(self, session_id, image_file, image_name, user_info=None):
        # log uploaded image and metadata
        timestamp = datetime.datetime.now().isoformat()
        
        # create session-specific uploads directory
        session_uploads_dir = self.log_dir / f"session_{session_id}" / "uploads"
        session_uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # save original image
        upload_path = session_uploads_dir / image_name
        
        # save the uploaded file
        if hasattr(image_file, 'getbuffer'):
            with open(upload_path, 'wb') as f:
                f.write(image_file.getbuffer())
        else:
            shutil.copy2(image_file, upload_path)
        
        # log metadata
        log_entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "action": "upload",
            "image_name": image_name,
            "image_path": str(upload_path),
            "user_info": user_info or "anonymous",
            "file_size": os.path.getsize(upload_path) if upload_path.exists() else 0
        }
        
        self._append_log(log_entry)
        return upload_path
    
    def log_analysis(self, session_id, parameters, results, analysis_duration=None):
        # log analysis parameters and complete results
        timestamp = datetime.datetime.now().isoformat()
        
        # save parameters
        params_file = self.log_dir / "parameters" / f"{session_id}_params.json"
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        # save results summary
        results_file = self.log_dir / "results" / f"{session_id}_results.json"
        
        # extract key metrics for logging
        colony_count = len(results['colony_properties']) if results and 'colony_properties' in results else 0
        
        results_summary = {
            "colony_count": colony_count,
            "has_morphology": 'morph_df' in results and not results['morph_df'].empty if results else False,
            "has_color_analysis": 'colony_data' in results if results else False,
            "has_density_analysis": 'density_df' in results and not results['density_df'].empty if results else False,
            "analysis_successful": results is not None
        }
        
        # save detailed results if successful
        if results:
            # create session-specific results directory
            session_results_dir = self.log_dir / f"session_{session_id}" / "results"
            session_results_dir.mkdir(parents=True, exist_ok=True)
            
            # save processed image
            if 'processed_image' in results:
                processed_img_path = session_results_dir / f"processed_image.png"
                from PIL import Image
                import numpy as np
                
                if isinstance(results['processed_image'], np.ndarray):
                    img = Image.fromarray(results['processed_image'])
                    img.save(processed_img_path)
            
            # save all available dataframes as CSV
            csv_files_saved = []
            
            # save colony data if available
            if 'combined_df' in results and not results['combined_df'].empty:
                csv_path = session_results_dir / "colonies_complete.csv"
                results['combined_df'].to_csv(csv_path, index=False)
                csv_files_saved.append(str(csv_path))
            
            # save morphology data
            if 'morph_df' in results and not results['morph_df'].empty:
                csv_path = session_results_dir / "morphology_analysis.csv"
                results['morph_df'].to_csv(csv_path, index=False)
                csv_files_saved.append(str(csv_path))
            
            # save color analysis data
            if 'colony_data' in results and results['colony_data']:
                csv_path = session_results_dir / "color_analysis.csv"
                import pandas as pd
                color_df = pd.DataFrame(results['colony_data'])
                color_df.to_csv(csv_path, index=False)
                csv_files_saved.append(str(csv_path))
            
            # save density analysis data
            if 'density_df' in results and not results['density_df'].empty:
                csv_path = session_results_dir / "density_analysis.csv"
                results['density_df'].to_csv(csv_path, index=False)
                csv_files_saved.append(str(csv_path))
            
            # save top colonies data
            if 'top_colonies_df' in results and not results['top_colonies_df'].empty:
                csv_path = session_results_dir / "top_colonies.csv"
                results['top_colonies_df'].to_csv(csv_path, index=False)
                csv_files_saved.append(str(csv_path))
            
            results_summary["csv_files"] = csv_files_saved
        
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # log session entry
        log_entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "action": "analysis",
            "parameters": parameters,
            "colony_count": colony_count,
            "analysis_duration": analysis_duration,
            "results_file": str(results_file),
            "params_file": str(params_file),
            "analysis_successful": results is not None
        }
        
        self._append_log(log_entry)
        self._update_summary()
    
    def log_download(self, session_id, download_type, filename):
        # log when users download results
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "action": "download",
            "download_type": download_type,
            "filename": filename
        }
        
        self._append_log(log_entry)
    
    def save_analysis_image(self, session_id, image_array, filename):
        # save analysis result images for admin viewing
        try:
            from PIL import Image
            import numpy as np
            
            # create session-specific results directory
            session_results_dir = self.log_dir / f"session_{session_id}" / "results"
            session_results_dir.mkdir(parents=True, exist_ok=True)
            
            # save image
            image_path = session_results_dir / filename
            
            if isinstance(image_array, np.ndarray):
                img = Image.fromarray(image_array.astype(np.uint8))
                img.save(image_path)
                return str(image_path)
            
        except Exception as e:
            print(f"Error saving analysis image: {e}")
            return None
    
    def _append_log(self, log_entry):
        # append entry to session log
        with open(self.session_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _update_summary(self):
        # create/update admin summary csv
        try:
            all_logs = []
            if self.session_log_file.exists():
                with open(self.session_log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            all_logs.append(json.loads(line))
            
            # create summary dataframe
            if all_logs:
                df = pd.DataFrame(all_logs)
                
                # group by session for summary
                summary_data = []
                for session_id in df['session_id'].unique():
                    session_logs = df[df['session_id'] == session_id]
                    
                    upload_logs = session_logs[session_logs['action'] == 'upload']
                    analysis_logs = session_logs[session_logs['action'] == 'analysis']
                    download_logs = session_logs[session_logs['action'] == 'download']
                    
                    if not upload_logs.empty:
                        first_upload = upload_logs.iloc[0]
                        summary_entry = {
                            'session_id': session_id,
                            'timestamp': first_upload['timestamp'],
                            'user_info': first_upload.get('user_info', 'anonymous'),
                            'image_name': first_upload['image_name'],
                            'total_uploads': len(upload_logs),
                            'total_analyses': len(analysis_logs),
                            'total_downloads': len(download_logs),
                            'colony_count': analysis_logs['colony_count'].iloc[-1] if not analysis_logs.empty else 0,
                            'analysis_successful': analysis_logs['analysis_successful'].iloc[-1] if not analysis_logs.empty else False
                        }
                        summary_data.append(summary_entry)
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_csv(self.summary_file, index=False)
        
        except Exception as e:
            print(f"error updating summary: {e}")
    
    def get_all_sessions(self):
        # return all session data for admin review
        if not self.summary_file.exists():
            return pd.DataFrame()
        
        return pd.read_csv(self.summary_file)
    
    def get_session_details(self, session_id):
        # get detailed data for specific session
        details = {}
        
        # get all logs for this session
        all_logs = []
        if self.session_log_file.exists():
            with open(self.session_log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        log = json.loads(line)
                        if log['session_id'] == session_id:
                            all_logs.append(log)
        
        details['logs'] = all_logs
        
        # get parameter file
        params_file = self.log_dir / "parameters" / f"{session_id}_params.json"
        if params_file.exists():
            with open(params_file, 'r') as f:
                details['parameters'] = json.load(f)
        
        # get results file
        results_file = self.log_dir / "results" / f"{session_id}_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                details['results_summary'] = json.load(f)
        
        # get colony data if available
        colony_file = self.log_dir / "results" / f"{session_id}_colonies.csv"
        if colony_file.exists():
            details['colony_data'] = pd.read_csv(colony_file)
        
        return details
    
    def export_all_data(self, output_dir="admin_export"):
        # export everything for admin review
        export_path = Path(output_dir)
        export_path.mkdir(exist_ok=True)
        
        # copy summary
        if self.summary_file.exists():
            shutil.copy2(self.summary_file, export_path / "admin_summary.csv")
        
        # copy session logs
        if self.session_log_file.exists():
            shutil.copy2(self.session_log_file, export_path / "session_log.jsonl")
        
        # create combined colony data
        all_colony_data = []
        for csv_file in (self.log_dir / "results").glob("*_colonies.csv"):
            session_id = csv_file.stem.replace("_colonies", "")
            df = pd.read_csv(csv_file)
            df['session_id'] = session_id
            all_colony_data.append(df)
        
        if all_colony_data:
            combined_df = pd.concat(all_colony_data, ignore_index=True)
            combined_df.to_csv(export_path / "all_colony_data.csv", index=False)
        
        # copy all images and results
        shutil.copytree(self.log_dir / "uploads", export_path / "uploads", dirs_exist_ok=True)
        shutil.copytree(self.log_dir / "processed_images", export_path / "processed_images", dirs_exist_ok=True)
        shutil.copytree(self.log_dir / "parameters", export_path / "parameters", dirs_exist_ok=True)
        
        return export_path

# global logger instance
admin_logger = AdminLogger() 