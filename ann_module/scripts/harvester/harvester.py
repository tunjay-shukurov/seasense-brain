from pathlib import Path
from datetime import datetime
from ann_module.scripts.harvester import fetcher, downloader
from ann_module.config import constants
from ann_module.scripts.harvester.channel import get_channel_list
from ann_module.config.user_params import get_user_params
from utils.logger import get_logger, log_harvester_event, log_harvester_run_start, log_harvester_run_end

def main():
    log = get_logger()
    
    # 1. Kullanıcı parametreleri
    params = get_user_params()
    
    # 2. Kanal listesi
    channel_list = get_channel_list(params.channel_type)
    if not channel_list:
        log.error(f"Unsupported channel type: {params.channel_type}")
        return
    
    # 3. Kayıt dizinleri ve etiket belirleme
    label = constants.LABEL_EARTHQUAKE if params.min_magnitude >= constants.MAG_THRESHOLD_EARTHQUAKE else constants.LABEL_NOISE
    
    if params.purpose == constants.PURPOSE_TEST:
        save_dir = constants.TEST_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        save_dir = constants.RAW_DIR / label / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. Log başlangıcı
    log_harvester_run_start(label, save_dir)
    log.info(f"Starting harvester run. Saving to {save_dir}")
    
    # 5. Olayları getir
    events = fetcher.fetch_events(params.starttime, params.endtime, params.min_magnitude, params.max_magnitude)
    if not events:
        log.warning("No events found for given parameters.")
        return
    
    downloaded_total = 0
    
    # 6. Olaylar üzerinde dön ve indir
    for idx, event in enumerate(events, 1):
        if downloaded_total >= params.target_download:
            break
        
        try:
            downloaded_count, stations, mag, timestamp, origin = downloader.download_event_streams(
                event, channel_list, save_dir, params.target_download, downloaded_total, label
            )
            
            downloaded_total += downloaded_count
            
            # Log harvester dosyasına detaylı kayıt
            log_harvester_event(
                event.resource_id.id.split("/")[-1],
                origin.time.strftime('%Y-%m-%d %H:%M:%S'),
                origin.latitude,
                origin.longitude,
                mag,
                downloaded_count,
                params.channel_type,
                label
            )
            
            # Terminal logu
            log.info(f"[{idx}/{len(events)}] Event {timestamp} - Mag: {mag:.1f} - SACs downloaded: {downloaded_count}")
        
        except Exception as e:
            log.error(f"Event skipped due to error: {e}")
    
    # 7. Fazla dosyaları temizle
    all_sac_files = sorted(save_dir.glob("*.SAC"))
    if len(all_sac_files) > params.target_download:
        for f in all_sac_files[params.target_download:]:
            f.unlink()
            log.info(f"Deleted excess file: {f.name}")
    
    # 8. Log sonu ve terminal bilgilendirme
    log_harvester_run_end(len(all_sac_files))
    log.info(f"Harvester run complete. Total SAC files downloaded: {len(all_sac_files)} → {save_dir}")

if __name__ == "__main__":
    main()
