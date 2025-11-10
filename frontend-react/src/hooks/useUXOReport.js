import { useState } from 'react';
import { apiService } from '../services/api';

/**
 * Hook tái sử dụng logic gửi báo cáo UXO
 * @param {string|null} token - admin token nếu có
 * @returns Các state và hàm điều khiển
 */
export const useUXOReport = (token = null) => {
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const submitReport = async () => {
    if (!selectedLocation) {
      throw new Error('Vui lòng chọn vị trí trên bản đồ trước khi gửi.');
    }
    if (!description.trim()) {
      throw new Error('Vui lòng thêm mô tả trước khi gửi.');
    }

    setIsSubmitting(true);
    try {
      await apiService.reportUXO(
        selectedLocation.lat,
        selectedLocation.lng,
        description,
        token
      );
      setSelectedLocation(null);
      setDescription('');
    } finally {
      setIsSubmitting(false);
    }
  };

  return {
    selectedLocation,
    setSelectedLocation,
    description,
    setDescription,
    isSubmitting,
    submitReport,
  };
};
