"""Analytics services for coaching session analysis."""

from app.services.analytics.alerts import AlertsService, get_alerts_service
from app.services.analytics.authorization import (
    require_coach_client_relationship,
    require_job_ownership,
    verify_coach_client_relationship,
    verify_job_ownership,
)
from app.services.analytics.coaching_style import (
    CoachingStyleAnalyzer,
    CoachingStyleMetrics,
)
from app.services.analytics.cue_detection import (
    CueDetectionError,
    CueDetectionService,
    get_cue_detection_service,
)
from app.services.analytics.engagement import (
    calculate_engagement_score,
    calculate_response_elaboration_score,
)
from app.services.analytics.risk_scoring import (
    RiskScoringService,
    get_risk_scoring_service,
    normalize_factor_contribution,
)
from app.services.analytics.sentiment import (
    aggregate_sentiment,
    calculate_sentiment_variance,
    sentiment_label_to_score,
)
from app.services.analytics.session_analytics import (
    SessionAnalyticsService,
    get_session_analytics_service,
)
from app.services.analytics.client_analytics import (
    ClientAnalyticsService,
    get_client_analytics_service,
)
from app.services.analytics.talk_time import TalkTimeAnalyzer, TalkTimeMetrics
from app.services.analytics.trends import (
    calculate_session_frequency_trend,
    calculate_talk_ratio_trend,
    calculate_trend,
)

__all__ = [
    # Talk-time
    "TalkTimeAnalyzer",
    "TalkTimeMetrics",
    # Coaching style
    "CoachingStyleAnalyzer",
    "CoachingStyleMetrics",
    # Cue detection
    "CueDetectionError",
    "CueDetectionService",
    "get_cue_detection_service",
    # Engagement
    "calculate_engagement_score",
    "calculate_response_elaboration_score",
    # Sentiment
    "aggregate_sentiment",
    "calculate_sentiment_variance",
    "sentiment_label_to_score",
    # Session analytics
    "SessionAnalyticsService",
    "get_session_analytics_service",
    # Client analytics
    "ClientAnalyticsService",
    "get_client_analytics_service",
    # Trends
    "calculate_trend",
    "calculate_talk_ratio_trend",
    "calculate_session_frequency_trend",
    # Risk scoring
    "RiskScoringService",
    "get_risk_scoring_service",
    "normalize_factor_contribution",
    # Alerts
    "AlertsService",
    "get_alerts_service",
    # Authorization
    "require_coach_client_relationship",
    "require_job_ownership",
    "verify_coach_client_relationship",
    "verify_job_ownership",
]
