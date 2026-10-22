class FraudDetectionError(Exception):
    pass


class ConfigurationError(FraudDetectionError):
    pass


class DataLoadingError(FraudDetectionError):
    pass


class DataValidationError(FraudDetectionError):
    pass


class DeviceUnavailableError(FraudDetectionError):
    pass


class TrainingError(FraudDetectionError):
    pass


class ArtifactError(FraudDetectionError):
    pass
